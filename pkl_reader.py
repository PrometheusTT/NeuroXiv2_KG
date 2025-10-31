import pickle
import argparse
import streamlit as st
import pandas as pd
import numpy as np
import types

# 小工具：把对象转成更可展示的摘要
def summarize(obj, maxlen=200):
    try:
        r = repr(obj)
    except Exception:
        r = f"<unrepr-able {type(obj)}>"
    if len(r) > maxlen:
        r = r[:maxlen] + "...(截断)"
    return r

def render_obj(name, obj, level=0):
    """
    递归展示任意 Python 对象
    """
    t = type(obj)

    # 基础类型 / 简单对象
    if isinstance(obj, (str, int, float, bool, type(None))):
        st.write(f"**{name}** ({t}): {obj}")
        return

    # pandas.DataFrame 特殊展示
    if "pandas" in t.__module__:
        if isinstance(obj, pd.DataFrame):
            with st.expander(f"{name}  [DataFrame] shape={obj.shape}", expanded=False):
                st.dataframe(obj.head(50))
        else:
            st.write(f"**{name}** ({t})")
        return

    # numpy.ndarray 特殊展示
    if "numpy" in t.__module__:
        if isinstance(obj, np.ndarray):
            with st.expander(f"{name}  [ndarray] shape={obj.shape} dtype={obj.dtype}", expanded=False):
                # 显示前几个值
                flat = obj.ravel()
                preview_vals = flat[: min(20, flat.size)]
                st.text(f"preview values (first {len(preview_vals)}): {preview_vals}")
        else:
            st.write(f"**{name}** ({t})")
        return

    # dict 展开
    if isinstance(obj, dict):
        with st.expander(f"{name}  [dict] size={len(obj)}", expanded=False):
            for k, v in list(obj.items()):  # 避免爆太大
                render_obj(f"[key] {k!r}", v, level+1)
        return

    # list / tuple 展开
    if isinstance(obj, (list, tuple)):
        with st.expander(f"{name}  [{t.__name__}] len={len(obj)}", expanded=False):
            for idx, v in enumerate(obj):
                render_obj(f"[{idx}]", v, level+1)
        return

    # 其他自定义类 / 复杂对象
    with st.expander(f"{name}  [{t}] ", expanded=False):
        # 尝试展示 __dict__ (对象的属性)
        if hasattr(obj, "__dict__"):
            st.write("属性字典 (__dict__)：")
            for k, v in list(vars(obj).items())[:200]:
                render_obj(f".{k}", v, level+1)
        else:
            # fallback: repr
            st.code(summarize(obj, maxlen=1000))

def main():

    # 载入
    with open('/home/wlj/NeuroXiv2/data/parc_r671_full.nrrd.pkl', "rb") as f:
        obj = pickle.load(f)

    st.sidebar.header("基本信息")
    st.sidebar.write("类型:", type(obj))
    if hasattr(obj, "__len__"):
        try:
            st.sidebar.write("长度:", len(obj))
        except Exception:
            pass

    # 主展示
    render_obj("root", obj)

if __name__ == "__main__":
    main()