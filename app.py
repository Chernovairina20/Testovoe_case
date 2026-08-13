# -*- coding: utf-8 -*-
"""
ЦифраФинанс — дашборд еженедельного мониторинга кредитного портфеля.
Запуск локально:  streamlit run app.py
Данные: data/loan_portfolio_scored.csv (очищенный + размеченный датасет),
        data/branch_reference.csv (плановые KPI по «регион-канал»).
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ЦифраФинанс — мониторинг портфеля", layout="wide")

DATA = os.path.join(os.path.dirname(__file__), "data")
TARGET_DR = 0.10  # плановый уровень дефолтов

EMP_RU = {"employed": "Наёмный", "self_employed": "Самозанятый",
          "freelance": "Фрилансер", "unemployed": "Безработный"}


@st.cache_data
def load():
    df = pd.read_csv(os.path.join(DATA, "loan_portfolio_scored.csv"), parse_dates=["issue_date"])
    ref = pd.read_csv(os.path.join(DATA, "branch_reference.csv"))
    return df, ref


df, ref = load()

# ------------------------------------------------------------------
# Фильтры
# ------------------------------------------------------------------
st.sidebar.header("Фильтры")
regions = st.sidebar.multiselect("Регион", sorted(df.region.unique()), sorted(df.region.unique()))
channels = st.sidebar.multiselect("Канал", sorted(df.channel.unique()), sorted(df.channel.unique()))
products = st.sidebar.multiselect("Продукт", sorted(df.loan_product.unique()), sorted(df.loan_product.unique()))
quarters = sorted(df.quarter.unique())
qsel = st.sidebar.select_slider("Период (кварталы)", options=quarters,
                                value=(quarters[0], quarters[-1]))

mask = (df.region.isin(regions) & df.channel.isin(channels) & df.loan_product.isin(products)
        & (df.quarter >= qsel[0]) & (df.quarter <= qsel[1]))
d = df[mask].copy()

st.title("ЦифраФинанс — мониторинг кредитного портфеля")
st.caption(f"Отобрано кредитов: {len(d):,} из {len(df):,}".replace(",", " "))

if d.empty:
    st.warning("Нет данных под выбранные фильтры.")
    st.stop()

# ------------------------------------------------------------------
# 4 KPI
# ------------------------------------------------------------------
vol = d.loan_amount.sum()
dr = d.is_default.mean()
el = d.EL_loan.sum()
avg_dti = d.dti_ratio.mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Объём портфеля", f"{vol/1e6:,.0f} млн ₽".replace(",", " "))
k2.metric("Default rate", f"{dr:.1%}", f"{(dr-TARGET_DR)*100:+.1f} пп к цели 10%",
          delta_color="inverse")
k3.metric("Expected Loss", f"{el/1e6:,.1f} млн ₽".replace(",", " "))
k4.metric("Средний DTI", f"{avg_dti:.1f}%")

st.divider()

# ------------------------------------------------------------------
# Двухосевой график: выдачи и дефолты по кварталам
# ------------------------------------------------------------------
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Динамика выдач и дефолтов по кварталам")
    q = d.groupby("quarter").agg(loans=("loan_id", "size"), dr=("is_default", "mean")).reset_index()
    fig = go.Figure()
    fig.add_bar(x=q.quarter, y=q.loans, name="Выдачи, шт.", marker_color="#a0aec0", yaxis="y1")
    fig.add_scatter(x=q.quarter, y=q.dr * 100, name="Default rate, %", mode="lines+markers",
                    line=dict(color="#c53030", width=3), yaxis="y2")
    fig.add_hline(y=10, line_dash="dot", line_color="#2f855a", yref="y2")
    fig.update_layout(
        yaxis=dict(title="Выдачи, шт."),
        yaxis2=dict(title="Default rate, %", overlaying="y", side="right", range=[0, 30]),
        legend=dict(orientation="h", y=1.15), height=380, margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Тепловая карта DR: канал × тип занятости
# ------------------------------------------------------------------
with c2:
    st.subheader("DR: канал × тип занятости")
    heat = (d.pivot_table("is_default", "channel", "employment_type", aggfunc="mean") * 100)
    heat = heat.rename(columns=EMP_RU)
    fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Reds",
                    labels=dict(color="DR, %"), aspect="auto")
    fig.update_layout(height=380, margin=dict(t=30, b=10), coloraxis_colorbar_title="DR, %")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Scatter: credit_score × DTI, цвет — дефолт
# ------------------------------------------------------------------
c3, c4 = st.columns([2, 3])
with c3:
    st.subheader("Скоринг × DTI (цвет — дефолт)")
    smp = d.sample(min(2500, len(d)), random_state=1).copy()
    smp["Статус"] = smp.is_default.map({0: "не дефолт", 1: "дефолт"})
    fig = px.scatter(smp, x="credit_score", y="dti_ratio", color="Статус",
                     color_discrete_map={"не дефолт": "#2b6cb0", "дефолт": "#c53030"},
                     opacity=0.45, labels={"credit_score": "Кредитный скоринг",
                                           "dti_ratio": "DTI, %", "Статус": ""})
    fig.add_vline(x=580, line_dash="dash", line_color="grey")
    fig.add_hline(y=65, line_dash="dash", line_color="grey")
    fig.update_layout(height=400, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Таблица план-факт по справочнику с подсветкой отклонений
# ------------------------------------------------------------------
with c4:
    st.subheader("План-факт по справочнику (регион × канал)")
    fact = d.groupby(["region", "channel"]).agg(
        Факт_DR=("is_default", "mean"),
        Факт_score=("credit_score", "mean"),
        Факт_DTI=("dti_ratio", "mean"),
        Кредитов=("loan_id", "size")).reset_index()
    pf = fact.merge(ref[["region", "channel", "target_default_rate",
                         "target_avg_credit_score", "target_avg_dti"]],
                    on=["region", "channel"], how="left")
    pf["Откл_DR_пп"] = (pf.Факт_DR - pf.target_default_rate) * 100
    pf["Откл_score"] = pf.Факт_score - pf.target_avg_credit_score
    pf["Откл_DTI"] = pf.Факт_DTI - pf.target_avg_dti
    pf["Факт_DR"] = (pf.Факт_DR * 100).round(1)
    pf["План_DR"] = (pf.target_default_rate * 100).round(1)
    pf["Факт_score"] = pf.Факт_score.round(0)
    pf["Факт_DTI"] = pf.Факт_DTI.round(1)
    show = pf[["region", "channel", "Кредитов", "Факт_DR", "План_DR", "Откл_DR_пп",
               "Откл_score", "Откл_DTI"]].copy()
    show["План_есть"] = np.where(pf.target_default_rate.notna(), "да", "нет плана")

    def hl(v):
        if pd.isna(v):
            return "color:#999"
        return "background-color:#fed7d7" if v > 0 else "background-color:#c6f6d5"

    sty = (show.style
           .map(hl, subset=["Откл_DR_пп", "Откл_DTI"])
           .format({"Откл_DR_пп": "{:+.1f}", "Откл_score": "{:+.0f}",
                    "Откл_DTI": "{:+.1f}", "Факт_DR": "{:.1f}", "План_DR": "{:.1f}"}, na_rep="—"))
    st.dataframe(sty, use_container_width=True, height=360)
    st.caption("Красный — факт хуже плана (выше по DR/DTI), зелёный — лучше. "
               "«нет плана» — срез не покрыт справочником (ограничение анализа).")
