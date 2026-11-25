"""FT-Transformer in TensorFlow/Keras 3, mirroring bin/ft_transformer.py of
yandex-research/tabular-dl-revisiting-models (numerical features only)."""
import math
import numpy as np
import tensorflow as tf
import keras
from keras import layers, initializers


def torch_kernel_init():
    # torch.nn.Linear weight init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    return initializers.VarianceScaling(scale=1.0 / 3.0, mode="fan_in", distribution="uniform")


def torch_bias_init(fan_in):
    b = 1.0 / math.sqrt(fan_in)
    return initializers.RandomUniform(-b, b)


def dense(units, fan_in, zero_bias=False):
    return layers.Dense(
        units,
        kernel_initializer=torch_kernel_init(),
        bias_initializer="zeros" if zero_bias else torch_bias_init(fan_in),
    )


class Tokenizer(layers.Layer):
    """[CLS] token + one embedding per numerical feature: x_j * W_j + b_j."""

    def __init__(self, d_numerical, d_token, bias=True, **kw):
        super().__init__(**kw)
        self.d_numerical, self.d_token, self.use_bias = d_numerical, d_token, bias

    def build(self, input_shape):
        b = 1.0 / math.sqrt(self.d_token)  # kaiming_uniform_(a=sqrt(5)) on (rows, d_token)
        init = initializers.RandomUniform(-b, b)
        self.weight = self.add_weight(shape=(self.d_numerical + 1, self.d_token), initializer=init, name="weight")
        self.bias = (
            self.add_weight(shape=(self.d_numerical, self.d_token), initializer=init, name="bias")
            if self.use_bias else None
        )

    def call(self, x_num):
        ones = tf.ones_like(x_num[:, :1])
        x = tf.concat([ones, x_num], axis=1)                    # [B, k+1]
        x = self.weight[None] * x[:, :, None]                   # [B, k+1, d]
        if self.bias is not None:
            bias = tf.concat([tf.zeros_like(self.bias[:1]), self.bias], axis=0)  # no bias for [CLS]
            x = x + bias[None]
        return x


class MultiheadAttention(layers.Layer):
    def __init__(self, d, n_heads, dropout, **kw):
        super().__init__(**kw)
        assert d % n_heads == 0 and n_heads > 1
        self.d, self.n_heads = d, n_heads
        self.W_q, self.W_k, self.W_v = (dense(d, d, zero_bias=True) for _ in range(3))
        self.W_out = dense(d, d, zero_bias=True)
        self.drop = layers.Dropout(dropout) if dropout else None

    def _split(self, x):
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.n_heads, self.d // self.n_heads])
        return tf.transpose(x, [0, 2, 1, 3])                    # [B, H, T, dh]

    def call(self, x_q, x_kv, training=False):
        q, k, v = self._split(self.W_q(x_q)), self._split(self.W_k(x_kv)), self._split(self.W_v(x_kv))
        dh = self.d // self.n_heads
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / math.sqrt(dh), axis=-1)
        if self.drop is not None:
            attn = self.drop(attn, training=training)
        x = tf.matmul(attn, v)                                  # [B, H, Tq, dh]
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.d])
        return self.W_out(x)


def reglu(x):
    a, b = tf.split(x, 2, axis=-1)
    return a * tf.nn.relu(b)


class FTTransformer(keras.Model):
    def __init__(self, d_numerical, d_out, n_layers, d_token, n_heads, d_ffn_factor,
                 attention_dropout, ffn_dropout, residual_dropout, prenormalization, **kw):
        super().__init__(**kw)
        self.prenorm, self.ffn_drop, self.res_drop = prenormalization, ffn_dropout, residual_dropout
        self.tokenizer = Tokenizer(d_numerical, d_token)
        d_hidden = int(d_token * d_ffn_factor)
        self.blocks = []
        for i in range(n_layers):
            blk = dict(
                attention=MultiheadAttention(d_token, n_heads, attention_dropout),
                linear0=dense(d_hidden * 2, d_token),           # ReGLU doubles the width
                linear1=dense(d_token, d_hidden),
                norm1=layers.LayerNormalization(epsilon=1e-5),
            )
            if not prenormalization or i:                       # first prenorm layer has no norm0
                blk["norm0"] = layers.LayerNormalization(epsilon=1e-5)
            self.blocks.append(blk)
        self.last_norm = layers.LayerNormalization(epsilon=1e-5) if prenormalization else None
        self.head = dense(d_out, d_token)
        self.d_out = d_out
        self.f_drop = layers.Dropout(ffn_dropout) if ffn_dropout else None
        self.r_drop = layers.Dropout(residual_dropout) if residual_dropout else None

    def _start(self, x, blk, key):
        return blk[key](x) if (self.prenorm and key in blk) else x

    def _end(self, x, res, blk, key, training):
        if self.r_drop is not None:
            res = self.r_drop(res, training=training)
        x = x + res
        return x if self.prenorm else blk[key](x)

    def call(self, x_num, training=False):
        x = self.tokenizer(x_num)
        for i, blk in enumerate(self.blocks):
            last = i + 1 == len(self.blocks)
            r = self._start(x, blk, "norm0")
            r = blk["attention"](r[:, :1] if last else r, r, training=training)  # last layer: [CLS] only
            if last:
                x = x[:, : tf.shape(r)[1]]
            x = self._end(x, r, blk, "norm0", training)
            r = self._start(x, blk, "norm1")
            r = reglu(blk["linear0"](r))
            if self.f_drop is not None:
                r = self.f_drop(r, training=training)
            r = blk["linear1"](r)
            x = self._end(x, r, blk, "norm1", training)
        x = x[:, 0]
        if self.last_norm is not None:
            x = self.last_norm(x)
        x = tf.nn.relu(x)                                       # last_activation for ReGLU
        x = self.head(x)
        return tf.squeeze(x, -1) if self.d_out == 1 else x


# ----------------------------------------------------------------------------
# Parameter naming (same names as the official PyTorch model) + weight-decay rule
# ----------------------------------------------------------------------------
def named_parameters(m):
    """[(torch_name, tf_variable, is_dense_kernel)] in the official PyTorch order."""
    P = [("tokenizer.weight", m.tokenizer.weight, False)]
    if m.tokenizer.bias is not None:
        P.append(("tokenizer.bias", m.tokenizer.bias, False))
    for i, b in enumerate(m.blocks):
        for k in ["W_q", "W_k", "W_v", "W_out"]:
            l = getattr(b["attention"], k)
            P += [(f"layers.{i}.attention.{k}.weight", l.kernel, True), (f"layers.{i}.attention.{k}.bias", l.bias, False)]
        for k in ["linear0", "linear1"]:
            P += [(f"layers.{i}.{k}.weight", b[k].kernel, True), (f"layers.{i}.{k}.bias", b[k].bias, False)]
        for k in ["norm0", "norm1"]:
            if k in b:
                P += [(f"layers.{i}.{k}.weight", b[k].gamma, False), (f"layers.{i}.{k}.bias", b[k].beta, False)]
    if m.last_norm is not None:
        P += [("last_normalization.weight", m.last_norm.gamma, False), ("last_normalization.bias", m.last_norm.beta, False)]
    P += [("head.weight", m.head.kernel, True), ("head.bias", m.head.bias, False)]
    return P


def needs_wd(name):
    # identical to the official code (note: 'last_normalization.weight' DOES get weight decay there)
    return all(x not in name for x in ["tokenizer", ".norm", ".bias"])


class TorchAdamW:
    """AdamW with exactly torch.optim.AdamW's update rule (decoupled decay, eps outside the bias correction)."""

    def __init__(self, variables, decay_flags, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.vars, self.lr, self.b1, self.b2, self.eps = variables, lr, betas[0], betas[1], eps
        self.wd = [weight_decay if f else 0.0 for f in decay_flags]
        self.m = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self.v = [tf.Variable(tf.zeros_like(v), trainable=False) for v in variables]
        self.t = tf.Variable(0.0, trainable=False)

    def apply(self, grads):
        self.t.assign_add(1.0)
        bc1 = 1.0 - tf.pow(self.b1, self.t)
        bc2 = 1.0 - tf.pow(self.b2, self.t)
        for p, g, m, v, wd in zip(self.vars, grads, self.m, self.v, self.wd):
            if g is None:
                continue
            g = tf.convert_to_tensor(g)
            p.assign(p * (1.0 - self.lr * wd))
            m.assign(self.b1 * m + (1.0 - self.b1) * g)
            v.assign(self.b2 * v + (1.0 - self.b2) * g * g)
            denom = tf.sqrt(v) / tf.sqrt(bc2) + self.eps
            p.assign_sub((self.lr / bc1) * m / denom)


# ----------------------------------------------------------------------------
# Data + training (mirrors lib/data.py:normalize and bin/ft_transformer.py's loop)
# ----------------------------------------------------------------------------
import json, time
from pathlib import Path
from scipy.special import expit


def load_dataset(dir_):
    """Loads N_{train,val,test}.npy / y_*.npy / info.json written in the official repo's format."""
    dir_ = Path(dir_)
    info = json.loads((dir_ / "info.json").read_text())
    N = {p: np.load(dir_ / f"N_{p}.npy") for p in ["train", "val", "test"]}
    y = {p: np.load(dir_ / f"y_{p}.npy") for p in ["train", "val", "test"]}
    assert info["n_cat_features"] == 0, "this port supports numerical features only (enough for Jannis / Higgs)"
    return N, y, info


def quantile_normalize(N, seed, noise=1e-3):
    """Same as lib.data.normalize(..., 'quantile', seed) incl. the noise trick for the fit."""
    from sklearn.preprocessing import QuantileTransformer
    N = {k: v.copy() for k, v in N.items()}
    if any(np.isnan(v).any() for v in N.values()):  # official: impute with the train mean
        mean = np.nanmean(N["train"], axis=0)
        for v in N.values():
            r, c = np.where(np.isnan(v)); v[r, c] = mean[c]
    X_train = N["train"].copy()
    normalizer = QuantileTransformer(
        output_distribution="normal", n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
        subsample=int(1e9), random_state=seed,
    )
    stds = np.std(X_train, axis=0, keepdims=True)
    noise_std = noise / np.maximum(stds, noise)
    X_train += noise_std * np.random.default_rng(seed).standard_normal(X_train.shape)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v).astype("float32") for k, v in N.items()}


def _score(task, y, logits):
    if task == "binclass":
        pred = np.round(expit(logits)).astype("int64")
    else:
        pred = logits.argmax(axis=1)
    return float((pred == y).mean())


def train_ft(N, y, info, model_cfg, train_cfg, seed, max_epochs=10**9, verbose=True):
    """Trains one FT-Transformer; returns metrics at the best-validation epoch."""
    assert model_cfg.get("activation", "reglu") == "reglu" and model_cfg.get("initialization", "kaiming") == "kaiming"
    assert not model_cfg.get("kv_compression")
    keras.utils.set_random_seed(seed)
    task = info["task_type"]; multiclass = task == "multiclass"
    d_out = info["n_classes"] if multiclass else 1
    X = quantile_normalize(N, seed)
    Xt = {p: tf.constant(X[p]) for p in X}
    yt_train = tf.constant(y["train"].astype("int64" if multiclass else "float32"))

    model = FTTransformer(
        d_numerical=X["train"].shape[1], d_out=d_out,
        **{k: model_cfg[k] for k in ["n_layers", "d_token", "n_heads", "d_ffn_factor", "attention_dropout",
                                     "ffn_dropout", "residual_dropout", "prenormalization"]},
    )
    model(Xt["train"][:2])
    P = named_parameters(model)
    variables = [v for _, v, _ in P]
    assert len(variables) == len(model.trainable_variables)
    n_params = int(sum(int(np.prod(v.shape)) for v in variables))
    opt = TorchAdamW(variables, [needs_wd(n) for n, _, _ in P], train_cfg["lr"], train_cfg["weight_decay"])
    bs, patience = train_cfg["batch_size"], train_cfg["patience"]
    n_train = X["train"].shape[0]

    def loss_fn(yb, logits):
        if multiclass:
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(yb, logits))
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yb, logits=logits))

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int32)])
    def train_step(idx):
        xb, yb = tf.gather(Xt["train"], idx), tf.gather(yt_train, idx)
        with tf.GradientTape() as tape:
            loss = loss_fn(yb, model(xb, training=True))
        opt.apply(tape.gradient(loss, variables))
        return loss

    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)

    def predict(part):
        x, out, eb = Xt[part], [], train_cfg.get("eval_batch_size", 8192)
        for i in range(0, x.shape[0], eb):
            out.append(infer(x[i:i + eb]).numpy())
        return np.concatenate(out)

    def evaluate():
        res = {}
        for part in ["val", "test"]:
            lg = predict(part); res[part] = _score(task, y[part], lg)
            if task == "binclass":
                from sklearn.metrics import roc_auc_score
                res[part + "_auc"] = float(roc_auc_score(y[part], expit(lg)))
        return res

    best, bad, best_epoch, best_w, best_res, t0 = -np.inf, 0, 0, None, None, time.time()
    for epoch in range(1, max_epochs + 1):
        perm = tf.random.shuffle(tf.range(n_train, dtype=tf.int32))
        losses = [train_step(perm[i:i + bs]) for i in range(0, n_train, bs)]
        res = evaluate()
        if res["val"] > best:                       # success (strict improvement), like zero.ProgressTracker
            best, bad, best_epoch, best_res = res["val"], 0, epoch, res
            best_w = [v.numpy() for v in variables]
        else:
            bad += 1
        if verbose:
            print(f"epoch {epoch:3d} | loss {float(tf.reduce_mean(losses)):.4f} | val {res['val']:.4f} | test {res['test']:.4f}"
                  f" | best val {best:.4f} (ep {best_epoch}) | {time.time() - t0:.0f}s", flush=True)
        if bad > patience:                          # patience + 1 consecutive epochs without improvement
            break
    for v, w in zip(variables, best_w):
        v.assign(w)
    final = evaluate()
    return {"seed": seed, "best_epoch": best_epoch, "val": final["val"], "test": final["test"],
            **({"test_auc": final["test_auc"]} if "test_auc" in final else {}),
            "n_parameters": n_params, "seconds": time.time() - t0}
