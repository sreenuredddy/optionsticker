import streamlit as st
import requests
import pandas as pd
import urllib3
import time
from streamlit_autorefresh import st_autorefresh
import streamlit as st
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Option Chain", layout="wide")
st.title("📈 Option Chain")
REFRESH_INTERVAL = 60  # seconds

col1, col2, col5 = st.columns(3)

with col1:
    # User input for expiry date
    expiry_input = st.text_input("📅 Enter Expiry Date (YYYY-MM-DD)", value="2025-08-14")
    # Choose index
    index = st.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY", "SENSEX"],
        index=0,
        help="Select an index or type a custom one",
        accept_new_options=True
    )

with col5:
    # Always show the manual refresh button on a new row
    if st.button("🔁 Manual Refresh"):
        st.session_state.last_refresh = time.time()
        st.rerun()

# Auto-refresh logic (only if toggle is ON)
auto_refresh = col2.toggle("Auto Refresh (1 min)", value=False)

if auto_refresh:
    st_autorefresh(interval=60 * 1000, key="auto_refresh")
    col2.markdown(f"🔄 **Last Refreshed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if expiry_input:
    # --- API CONFIG ---
    # Construct API URL dynamically
    API_URL = (f"https://nw.nuvamawealth.com/edelmw-content/"f"content/new-options/option-chain/NFO/OPTIDX/{index}/{expiry_input}?screen=all")

    HEADERS = {
        "Content-Type": "application/json",
        "Appidkey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHAiOjEsImZmIjoiVyIsImJkIjoid2ViLXBjIiwibmJmIjoxNTc5MjQxODMyLCJzcmMiOiJlbXRtdyIsImF2IjoiMS4wLjAuNCIsImFwcGlkIjoiNGZlNjhiNzUzNjc4NGUzNDA3YzNlY2YxOWJlN2M0YWQiLCJpc3MiOiJlbXQiLCJleHAiOjE2MTA3NzgxMzIsImlhdCI6MTU3OTI0MjEzMn0.IR-PKf1Jjr69bsERFmMeuZrZ2RafBDiTGgKA6Ygofdo",
        "Source": "EDEL",
        "Origin": "https://www.nuvamawealth.com",
        "Referer": "https://www.nuvamawealth.com/",
        "User-Agent": "Mozilla/5.0"
    }

    try:
        # --- FETCH NIFTY LTP & CHANGE ---
        response = requests.get(API_URL, headers=HEADERS, verify=False, timeout=20)
        data = response.json()
        nifty_value = data.get("data", {}).get("spObj", [])
        nifty_ltp = round(float(nifty_value.get("ltp", 0)), 2)
        nifty_chg = round(float(nifty_value.get("chg", 0)), 2)
        nifty_chg_pct = round(float(nifty_value.get("chgP", 0)), 2)

        op_chain = data.get("data", {}).get("opChn", [])

        if not op_chain:
            st.warning("⚠️ No option chain data available.")
            st.stop()

        rows = []

        for item in op_chain:
            ce = item.get("ceQt", {})
            pe = item.get("peQt", {})
            strike = int(round(float(item.get("stkPrc", 0))))

            ce_ltp = float(ce.get("ltp", 0))
            pe_ltp = float(pe.get("ltp", 0))
            ce_oi = float(ce.get("opInt", 0))
            pe_oi = float(pe.get("opInt", 0))
            ce_coi = float(ce.get("opIntChg", 0))
            pe_coi = float(pe.get("opIntChg", 0))
            ce_vol = float(ce.get("vol", 0))
            pe_vol = float(pe.get("vol", 0))
            ce_chgp = float(ce.get("chgP", 0))
            pe_chgp = float(pe.get("chgP", 0))

            ce_intrinsic = max(nifty_ltp - strike, 0)
            pe_intrinsic = max(strike - nifty_ltp, 0)
            ce_extrinsic = ce_ltp - ce_intrinsic
            pe_extrinsic = pe_ltp - pe_intrinsic

            
            # --- Trend Logic ---
            try:
                pcr_oi_val = pe_oi / ce_oi if ce_oi else None
                pcr_coi_val = pe_coi / ce_coi if ce_coi else None
                pcr_vol_val = pe_vol / ce_vol if ce_vol else None

                if all(pcr is not None and pcr > 1 for pcr in [pcr_oi_val, pcr_coi_val, pcr_vol_val]):
                    trend = "Bull"
                elif all(pcr is not None and pcr < 1 for pcr in [pcr_oi_val, pcr_coi_val, pcr_vol_val]):
                    trend = "Bear"
                else:
                    trend = "Neutral"
            except ZeroDivisionError:
                trend = "Neutral"

            rows.append({
                "CE Delta": ce.get("delta", ""),
                "CE Int": f"{ce_intrinsic:.2f}",
                "CE Ext": f"{ce_extrinsic:.2f}",
                "CE Br": f"{round(strike + ce_ltp, 2):.2f}",
                "CE (OI/Chg)": f"{ce_oi:.0f} / {ce_coi:.0f}",
                "CE LTP": f"{ce_ltp:.2f}",
                "CE Chgp": f"{ce_chgp:.2f}",
                "Strike": strike,
                "vtySkw": f"{float(item.get('vtySkw', 0)):.2f}",
                "PE LTP": f"{pe_ltp:.2f} ",
                "PE Chgp": f"{pe_chgp:.2f}",
                "PE (OI/Chg)": f"{pe_oi:.0f} / {pe_coi:.0f}",
                "PE Br": f"{round(strike - pe_ltp, 2):.2f}",
                "PE Ext": f"{pe_extrinsic:.2f}",
                "PE Int": f"{pe_intrinsic:.2f}",
                "PE Delta": pe.get("delta", ""),
                "Trend" : trend,
                # "optPn": f"{float(item.get('optPn', 0)):.2f}",
                "pcr": f"{float(item.get('pcr', 0)):.2f}",
                "PCR COI": f"{round(pe_coi / ce_coi, 2):.2f}" if ce_coi else None,
                "PCR Vol": f"{round(pe_vol / ce_vol, 2):.2f}" if ce_vol else None,

                # --- CE Side ---
                #"CE LTP": ce.get("ltp", ""),
                "CE Change": ce.get("chg", ""),
                "CE Chg%": ce.get("chgP", ""),
                "CE OI": ce.get("opInt", ""),
                "CE OI Chg": ce.get("opIntChg", ""),
                "CE OI Chg%": ce.get("opIntChgP", ""),
                "CE prcOIA": ce.get("prcOIA", ""),
                "CE Ask Price": ce.get("askPr", ""),
                "CE Ask Size": ce.get("akSz", ""),
                "CE Bid Price": ce.get("bidPr", ""),
                "CE Bid Size": ce.get("bdSz", ""),
                "CE IV Fut @ LTP": ce.get("ltpivfut", ""),
                "CE IV Spot @ LTP": ce.get("ltpivspt", ""),
                "CE IV Fut @ Ask": ce.get("askivfut", ""),
                "CE IV Fut @ Bid": ce.get("bidivfut", ""),
                
                "CE Gamma": ce.get("gamma", ""),
                "CE Theta": ce.get("theta", ""),
                "CE Vega": ce.get("vega", ""),
                "CE Open": ce.get("o", ""),
                "CE High": ce.get("h", ""),
                "CE Low": ce.get("l", ""),
                "CE Prev Close": ce.get("c", ""),
                "CE Volume": ce.get("vol", ""),
                "CE Trading Symbol": ce.get("trdSym", ""),
                "CE Symbol": ce.get("sym", ""),
                

                # --- PE Side ---
                #"PE LTP": pe.get("ltp", ""),
                "PE Change": pe.get("chg", ""),
                "PE Chg%": pe.get("chgP", ""),
                "PE OI": pe.get("opInt", ""),
                "PE OI Chg": pe.get("opIntChg", ""),
                "PE OI Chg%": pe.get("opIntChgP", ""),
                "PE prcOIA": pe.get("prcOIA", ""),
                "PE Ask Price": pe.get("askPr", ""),
                "PE Ask Size": pe.get("akSz", ""),
                "PE Bid Price": pe.get("bidPr", ""),
                "PE Bid Size": pe.get("bdSz", ""),
                "PE IV Fut @ LTP": pe.get("ltpivfut", ""),
                "PE IV Spot @ LTP": pe.get("ltpivspt", ""),
                "PE IV Fut @ Ask": pe.get("askivfut", ""),
                "PE IV Fut @ Bid": pe.get("bidivfut", ""),
                
                "PE Gamma": pe.get("gamma", ""),
                "PE Theta": pe.get("theta", ""),
                "PE Vega": pe.get("vega", ""),
                "PE Open": pe.get("o", ""),
                "PE High": pe.get("h", ""),
                "PE Low": pe.get("l", ""),
                "PE Prev Close": pe.get("c", ""),
                "PE Volume": pe.get("vol", ""),
                "PE Trading Symbol": pe.get("trdSym", ""),
                "PE Symbol": pe.get("sym", ""),

            })

        df = pd.DataFrame(rows).sort_values("Strike")

        # --- STRIKE PRICE FOCUS ---

        available_strikes = sorted(df["Strike"].unique())
        prev_close = round(nifty_ltp - nifty_chg, 2)
        nearest_strike = min(available_strikes, key=lambda x: abs(x - prev_close))
        max_pain_strike = data.get("data", {}).get("maxPn", None)
        nifty_pcr = data.get("data", {}).get("pcr", None)

        col3, col4, col6 = st.columns(3)

        with col3:
            st.markdown(f"""
            - 🟢 Nifty 50 LTP: `{nifty_ltp:.2f}`
            - 🔄 Change: `{nifty_chg:.2f}` ({nifty_chg_pct:.2f}%)
            - ⚡ PCR: `{nifty_pcr}`
            """)
            num_strikes = st.slider("Number of strikes above/below", min_value=1, max_value=20, value=10)

        with col4:
            st.markdown(f"""
            - 🎯 Previous Close: `{prev_close:.2f}`
            - 🧲 Closest Strike: `{nearest_strike}`
            - ⚡ Max Pain: `{max_pain_strike}`
            """)

        user_strike = nearest_strike
        center_index = available_strikes.index(user_strike)
        start = max(center_index - num_strikes, 0)
        end = min(center_index + num_strikes + 1, len(available_strikes))
        selected_strikes = available_strikes[start:end]

        filtered_df = df[df["Strike"].isin(selected_strikes)].copy()

        def highlight_center(row):
            color = "background-color: black; color: white; font-weight: bold;" if row["Strike"] == user_strike else ""
            return [color] * len(row)
        def highlight_strike_column(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            styles['Strike'] = 'background-color: black; color: white; font-weight: bold;'
            #styles['Trend'] = 'background-color: white; color: black; font-weight: bold;'
            return styles
        def zebra_stripes(row):
            return ['background-color: #f2f2f7; font-weight: bold;color: black; ' if row.name % 2 == 0 else 'background-color: #ffffff; font-weight: bold;color: black; ' for _ in row]
        # Find strike closest to the current LTP
        ltp_closest_strike = min(available_strikes, key=lambda x: abs(x - nifty_ltp))

        def highlight_ltp_strike(row):
            if row["Strike"] == ltp_closest_strike:
                return ["background-color: aqua; color: black; font-weight: bold;"] * len(row)
            return [""] * len(row)
        # Remove default index so no sequence numbers show up
        filtered_df = filtered_df.reset_index(drop=True)

        # Identify highest OI and OI change for CE and PE
        max_ce_oi = filtered_df["CE (OI/Chg)"].apply(lambda x: float(x.split(" / ")[0])).max()
        max_ce_coi = filtered_df["CE (OI/Chg)"].apply(lambda x: float(x.split(" / ")[1])).max()
        max_pe_oi = filtered_df["PE (OI/Chg)"].apply(lambda x: float(x.split(" / ")[0])).max()
        max_pe_coi = filtered_df["PE (OI/Chg)"].apply(lambda x: float(x.split(" / ")[1])).max()

        def highlight_max_oi(row):
            styles = [""] * len(row)
            ce_oi, ce_coi = map(float, row["CE (OI/Chg)"].split(" / "))
            pe_oi, pe_coi = map(float, row["PE (OI/Chg)"].split(" / "))

            # Highlight CE OI
            if ce_oi == max_ce_oi:
                styles[row.index.get_loc("CE (OI/Chg)")] = "background-color: red; font-weight: bold; color: black;"
            # Highlight CE OI Change
            if ce_coi == max_ce_coi:
                styles[row.index.get_loc("CE (OI/Chg)")] = "background-color: #ff9999; font-weight: bold; color: white;"

            # Highlight PE OI
            if pe_oi == max_pe_oi:
                styles[row.index.get_loc("PE (OI/Chg)")] = "background-color: green; font-weight: bold; color: black;"
            # Highlight PE OI Change
            if pe_coi == max_pe_coi:
                styles[row.index.get_loc("PE (OI/Chg)")] = "background-color: lightgreen; font-weight: bold; color: white;"

            return styles

        styled_df = (
            filtered_df.style
            .hide(axis="index")
            .apply(zebra_stripes, axis=1)
            .apply(highlight_center, axis=1)
            .apply(highlight_strike_column, axis=None)
            .apply(highlight_ltp_strike, axis=1)
            .apply(highlight_max_oi, axis=1)  # 👈 NEW highlighting for OI
        )

        # --- Merge CE + PE into one row ---
        ce_df = filtered_df[["Strike", "vtySkw", "CE OI Chg", "CE Chg%"]].copy()
        ce_df.rename(columns={"CE OI Chg": "CE COI", "CE Chg%": "CE Price Chg"}, inplace=True)

        pe_df = filtered_df[["Strike", "PE OI Chg", "PE Chg%"]].copy()
        pe_df.rename(columns={"PE OI Chg": "PE COI", "PE Chg%": "PE Price Chg"}, inplace=True)

        # Merge on Strike
        df = pd.merge(ce_df, pe_df, on="Strike", how="inner")

        # --- Convert to numeric ---
        for col in ["vtySkw", "CE COI", "CE Price Chg", "PE COI", "PE Price Chg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)

        # Instead, map API fields directly (assuming you have access to them here)
        df["CE Classification"] = filtered_df["CE prcOIA"].astype(str)
        df["PE Classification"] = filtered_df["PE prcOIA"].astype(str)

        # When determining dominant side, and further logic, use these columns
        top = df.loc[df["vtySkw"].idxmax()]
        top_strike = int(top["Strike"])
        top_skew = float(top["vtySkw"])

        dominant = None
        if abs(top["CE COI"]) >= abs(top["PE COI"]):
            dominant = f"CE: {top['CE Classification']}"
        else:
            dominant = f"PE: {top['PE Classification']}"

        # For display in your plot labels and dataframe, replace the Skew Label generation to include CE/PE Classification from API:

        def make_label(side, classification, coi, price_chg, other_side, other_classification, other_coi, other_price):
            return (
                f"{classification}<br>"
                f"{side} COI: {coi:+,.0f}, Price Chg: {price_chg:+.2f}%<br>"
                f"{other_side} COI: {other_coi:+,.0f}, Price Chg: {other_price:+.2f}%"
            )

        df["Skew Label"] = df.apply(
            lambda r: make_label(
                "CE", r["CE Classification"], r["CE COI"], r["CE Price Chg"],
                "PE", r["PE Classification"], r["PE COI"], r["PE Price Chg"]
            ),
            axis=1
        )

        # --- Plot ---
        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Strike"], y=df["vtySkw"], mode="lines+markers+text",
            name="Volatility Skew",
            text=df["Skew Label"],
            textposition="top center",
            hovertemplate="%{text}<extra></extra>"
        ))

        fig.update_layout(
            title="Volatility Skew with Buyer/Seller Context + Dominant OI",
            xaxis_title="Strike",
            yaxis_title="vtySkw",
            hovermode="x unified"
        )
        with col6:
            # --- Directional recommendation ---
            st.markdown(f"""
            - Top Skew Strike:`{top_strike}`
            - vtySkw:`{top_skew:.2f}`
            - Dominant Driver:`{dominant}`
            """)


        st.plotly_chart(fig, use_container_width=True)
        # --- Display table ---
        #st.dataframe(df.sort_values("vtySkw", ascending=False), use_container_width=True)

        #Highest volume
        # After you have 'filtered_df' DataFrame ready from your existing code
        # Convert CE and PE volumes to numeric for proper comparison
        filtered_df["CE Volume"] = pd.to_numeric(filtered_df["CE Volume"], errors="coerce")
        filtered_df["PE Volume"] = pd.to_numeric(filtered_df["PE Volume"], errors="coerce")

        # Get top 3 CE and rename column
        top3_ce_vol_strikes = (
            filtered_df.nlargest(3, "CE Volume")[["Strike", "CE Volume", "CE prcOIA", "CE LTP", "CE (OI/Chg)"]]
            .rename(columns={"CE prcOIA": "CE Interpretation"})
        )
        top3_ce_vol_strikes.index = range(1, len(top3_ce_vol_strikes) + 1)

        # Get top 3 PE and rename column
        top3_pe_vol_strikes = (
            filtered_df.nlargest(3, "PE Volume")[["Strike", "PE Volume", "PE prcOIA", "PE LTP", "PE (OI/Chg)"]]
            .rename(columns={"PE prcOIA": "PE Interpretation"})
        )
        top3_pe_vol_strikes.index = range(1, len(top3_pe_vol_strikes) + 1)

        # Display results without index column
        st.markdown("## Top 3 Highest Volume Strikes")

        st.markdown("### Call Options (CE) with Highest Volumes")
        st.dataframe(top3_ce_vol_strikes.style.hide(axis="index"), use_container_width=True)

        st.markdown("### Put Options (PE) with Highest Volumes")
        st.dataframe(top3_pe_vol_strikes.style.hide(axis="index"), use_container_width=True)

        # =======================
            
        # --- Ensure numeric conversion for OI, OI Change, and Price Change columns ---
        cols = ["CE OI", "PE OI", "CE OI Chg", "PE OI Chg", "CE Chg%", "PE Chg%"]
        for col in cols:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

        # ===============================
        # 📌 Identify highest metrics
        # ===============================
        max_ce_oi = filtered_df["CE OI"].max()
        max_ce_oi_strikes = filtered_df.loc[filtered_df["CE OI"] == max_ce_oi, "Strike"].tolist()
        max_ce_oi_price_chg = filtered_df.loc[filtered_df["CE OI"] == max_ce_oi, "CE Chg%"].iloc[0] if max_ce_oi_strikes else None

        max_pe_oi = filtered_df["PE OI"].max()
        max_pe_oi_strikes = filtered_df.loc[filtered_df["PE OI"] == max_pe_oi, "Strike"].tolist()
        max_pe_oi_price_chg = filtered_df.loc[filtered_df["PE OI"] == max_pe_oi, "PE Chg%"].iloc[0] if max_pe_oi_strikes else None

        # Absolute OI Change
        max_ce_oi_chg_abs = filtered_df["CE OI Chg"].abs().max()
        max_ce_oi_chg_abs_strikes = filtered_df.loc[filtered_df["CE OI Chg"].abs() == max_ce_oi_chg_abs, "Strike"].tolist()
        actual_ce_oi_chg_abs_val = filtered_df.loc[filtered_df["CE OI Chg"].abs() == max_ce_oi_chg_abs, "CE OI Chg"].iloc[0]
        max_ce_oi_chg_abs_price = filtered_df.loc[filtered_df["CE OI Chg"].abs() == max_ce_oi_chg_abs, "CE Chg%"].iloc[0]

        max_pe_oi_chg_abs = filtered_df["PE OI Chg"].abs().max()
        max_pe_oi_chg_abs_strikes = filtered_df.loc[filtered_df["PE OI Chg"].abs() == max_pe_oi_chg_abs, "Strike"].tolist()
        actual_pe_oi_chg_abs_val = filtered_df.loc[filtered_df["PE OI Chg"].abs() == max_pe_oi_chg_abs, "PE OI Chg"].iloc[0]
        max_pe_oi_chg_abs_price = filtered_df.loc[filtered_df["PE OI Chg"].abs() == max_pe_oi_chg_abs, "PE Chg%"].iloc[0]

        # Positive OI Change
        max_ce_oi_chg_pos = filtered_df["CE OI Chg"].max()
        max_ce_oi_chg_pos_strikes = filtered_df.loc[filtered_df["CE OI Chg"] == max_ce_oi_chg_pos, "Strike"].tolist()
        max_ce_oi_chg_pos_price = filtered_df.loc[filtered_df["CE OI Chg"] == max_ce_oi_chg_pos, "CE Chg%"].iloc[0]

        max_pe_oi_chg_pos = filtered_df["PE OI Chg"].max()
        max_pe_oi_chg_pos_strikes = filtered_df.loc[filtered_df["PE OI Chg"] == max_pe_oi_chg_pos, "Strike"].tolist()
        max_pe_oi_chg_pos_price = filtered_df.loc[filtered_df["PE OI Chg"] == max_pe_oi_chg_pos, "PE Chg%"].iloc[0]

        # Negative OI Change
        max_ce_oi_chg_neg = filtered_df["CE OI Chg"].min()
        max_ce_oi_chg_neg_strikes = filtered_df.loc[filtered_df["CE OI Chg"] == max_ce_oi_chg_neg, "Strike"].tolist()
        max_ce_oi_chg_neg_price = filtered_df.loc[filtered_df["CE OI Chg"] == max_ce_oi_chg_neg, "CE Chg%"].iloc[0]

        max_pe_oi_chg_neg = filtered_df["PE OI Chg"].min()
        max_pe_oi_chg_neg_strikes = filtered_df.loc[filtered_df["PE OI Chg"] == max_pe_oi_chg_neg, "Strike"].tolist()
        max_pe_oi_chg_neg_price = filtered_df.loc[filtered_df["PE OI Chg"] == max_pe_oi_chg_neg, "PE Chg%"].iloc[0]

        # ===============================
        # 📌 Permanent Shift Tracker in Two Columns
        # ===============================
        if "shift_tracker" not in st.session_state:
            st.session_state["shift_tracker"] = {}

        def update_shift(key, old_value, new_value, val, price):
            """Update tracker for single metric"""
            if old_value != new_value:
                st.session_state["shift_tracker"][key] = f"{old_value} → {new_value} (Value: {val:,.0f}, Price Chg: {price:+.2f}%)"
            elif key not in st.session_state["shift_tracker"]:
                st.session_state["shift_tracker"][key] = f"{new_value} (Value: {val:,.0f}, Price Chg: {price:+.2f}%)"

        # Update each metric
        update_shift("🚨 OI Call", st.session_state.get("last_oi_call"), max_ce_oi_strikes, max_ce_oi, max_ce_oi_price_chg)
        update_shift("🚨 OI Put", st.session_state.get("last_oi_put"), max_pe_oi_strikes, max_pe_oi, max_pe_oi_price_chg)
        update_shift("Abs OI Change Call", st.session_state.get("last_abs_call"), max_ce_oi_chg_abs_strikes, actual_ce_oi_chg_abs_val, max_ce_oi_chg_abs_price)
        update_shift("Abs OI Change Put", st.session_state.get("last_abs_put"), max_pe_oi_chg_abs_strikes, actual_pe_oi_chg_abs_val, max_pe_oi_chg_abs_price)
        update_shift("+OI Change Call", st.session_state.get("last_pos_call"), max_ce_oi_chg_pos_strikes, max_ce_oi_chg_pos, max_ce_oi_chg_pos_price)
        update_shift("+OI Change Put", st.session_state.get("last_pos_put"), max_pe_oi_chg_pos_strikes, max_pe_oi_chg_pos, max_pe_oi_chg_pos_price)
        update_shift("-OI Change Call", st.session_state.get("last_neg_call"), max_ce_oi_chg_neg_strikes, max_ce_oi_chg_neg, max_ce_oi_chg_neg_price)
        update_shift("-OI Change Put", st.session_state.get("last_neg_put"), max_pe_oi_chg_neg_strikes, max_pe_oi_chg_neg, max_pe_oi_chg_neg_price)

        # Save current values for next run comparisons
        st.session_state["last_oi_call"] = max_ce_oi_strikes
        st.session_state["last_oi_put"] = max_pe_oi_strikes
        st.session_state["last_abs_call"] = max_ce_oi_chg_abs_strikes
        st.session_state["last_abs_put"] = max_pe_oi_chg_abs_strikes
        st.session_state["last_pos_call"] = max_ce_oi_chg_pos_strikes
        st.session_state["last_pos_put"] = max_pe_oi_chg_pos_strikes
        st.session_state["last_neg_call"] = max_ce_oi_chg_neg_strikes
        st.session_state["last_neg_put"] = max_pe_oi_chg_neg_strikes

        # ===============================
        # 📌 Display Shift Updates in Two Columns
        # ===============================
        st.markdown("# 📢 Shift Updates ")

        col1, col2 = st.columns(2)

        col1_keys = ["🚨 OI Call", "🚨 OI Put", "Abs OI Change Call", "Abs OI Change Put"]
        col2_keys = ["+OI Change Call", "+OI Change Put", "-OI Change Call", "-OI Change Put"]

        with col1:
            for key in col1_keys:
                if key in st.session_state["shift_tracker"]:
                    st.markdown(f"- **{key}:** {st.session_state['shift_tracker'][key]}")

        with col2:
            for key in col2_keys:
                if key in st.session_state["shift_tracker"]:
                    st.markdown(f"- **{key}:** {st.session_state['shift_tracker'][key]}")

        # New Money

        # ==========================
        # 📊 NEW MONEY FLOW FUNCTION
        # ==========================
        def show_new_money_flow_overall_split(filtered_df, top_n=None, strike_col="Strike"):
            # --- Ensure numeric columns ---
            filtered_df["CE OI"] = pd.to_numeric(filtered_df["CE OI"], errors="coerce")
            filtered_df["PE OI"] = pd.to_numeric(filtered_df["PE OI"], errors="coerce")
            filtered_df["CE OI Chg"] = pd.to_numeric(filtered_df["CE OI Chg"], errors="coerce")
            filtered_df["PE OI Chg"] = pd.to_numeric(filtered_df["PE OI Chg"], errors="coerce")
            filtered_df["CE Chg%"] = pd.to_numeric(filtered_df["CE Chg%"], errors="coerce")
            filtered_df["PE Chg%"] = pd.to_numeric(filtered_df["PE Chg%"], errors="coerce")

            # ---- Prepare CE table ----
            ce_df = filtered_df[[strike_col, "CE OI", "CE OI Chg", "CE Chg%", "CE prcOIA"]].copy()
            ce_df = ce_df.rename(columns={
                "CE OI": "OI",
                "CE OI Chg": "OI Chg",
                "CE Chg%": "Chg%",
                "CE prcOIA": "Interpretation"
            })
            ce_df = ce_df.sort_values("OI Chg", ascending=False)
            if top_n:
                ce_df = ce_df.head(top_n)

            # ---- Prepare PE table ----
            pe_df = filtered_df[[strike_col, "PE OI", "PE OI Chg", "PE Chg%", "PE prcOIA"]].copy()
            pe_df = pe_df.rename(columns={
                "PE OI": "OI",
                "PE OI Chg": "OI Chg",
                "PE Chg%": "Chg%",
                "PE prcOIA": "Interpretation"
            })
            pe_df = pe_df.sort_values("OI Chg", ascending=False)
            if top_n:
                pe_df = pe_df.head(top_n)

            # ---- Display side-by-side ----
            st.subheader("📊 New Money Flow - Overall (Separate for CE & PE)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🟢 Call Options (CE)")
                st.dataframe(ce_df.reset_index(drop=True), use_container_width=True)
            with col2:
                st.markdown("### 🔴 Put Options (PE)")
                st.dataframe(pe_df.reset_index(drop=True), use_container_width=True)

            # ---- Totals ----
            total_ce_all_oi = filtered_df["CE OI"].sum()
            total_pe_all_oi = filtered_df["PE OI"].sum()
            total_ce_all_chg = filtered_df["CE OI Chg"].sum()
            total_pe_all_chg = filtered_df["PE OI Chg"].sum()

            # ---- Summary ----
            st.markdown("## 📌 OI & OI Change Summary")
            scol1, scol2 = st.columns(2)
            with scol1:
                st.markdown(f"**Total CE OI (All Strikes)** `{total_ce_all_oi:,.0f}`")
                st.markdown(f"**Total CE OI Change (All Strikes)** `{total_ce_all_chg:+,.0f}`")
            with scol2:
                st.markdown(f"**Total PE OI (All Strikes)** `{total_pe_all_oi:,.0f}`")
                st.markdown(f"**Total PE OI Change (All Strikes)** `{total_pe_all_chg:+,.0f}`")


        # ==========================
        # 📌 CALL FUNCTION
        # ==========================
        # Show top 10 New Money Flow rows by default
        show_new_money_flow_overall_split(filtered_df, top_n=10)

                #Straddle
        import plotly.graph_objects as go

        # Prepare data for straddle chart (average of Call and Put prices)
        strikes = filtered_df["Strike"]
        ce_prev_close = pd.to_numeric(filtered_df["CE Prev Close"], errors="coerce")
        pe_prev_close = pd.to_numeric(filtered_df["PE Prev Close"], errors="coerce")
        straddle_prev_close = (ce_prev_close + pe_prev_close) 

        ce_open = pd.to_numeric(filtered_df["CE Open"], errors="coerce")
        pe_open = pd.to_numeric(filtered_df["PE Open"], errors="coerce")
        straddle_open = (ce_open + pe_open) 

        ce_ltp = pd.to_numeric(filtered_df["CE LTP"], errors="coerce")
        pe_ltp = pd.to_numeric(filtered_df["PE LTP"], errors="coerce")
        straddle_ltp = (ce_ltp + pe_ltp)

        fig_straddle = go.Figure()

        fig_straddle.add_trace(go.Scatter(
            x=strikes,
            y=straddle_prev_close,
            mode="lines+markers",
            name="Straddle Prev Close"
        ))

        fig_straddle.add_trace(go.Scatter(
            x=strikes,
            y=straddle_open,
            mode="lines+markers",
            name="Straddle Open"
        ))

        fig_straddle.add_trace(go.Scatter(
            x=strikes,
            y=straddle_ltp,
            mode="lines+markers",
            name="Straddle LTP"
        ))

        fig_straddle.update_layout(
            title="Straddle Chart (Mean of CE and PE) - Prev Close, Open, LTP",
            xaxis_title="Strike Price",
            yaxis_title="Price",
            hovermode="x unified"
        )

        st.plotly_chart(fig_straddle, use_container_width=True)
        
        # Assuming filtered_df has these columns: "Strike", "Trend"
        atm_price = nifty_ltp  # Current underlying price (Nifty LTP)

        # Filter for Bull and Bear trends separately
        bull_strikes = filtered_df[filtered_df["Trend"] == "Bull"]["Strike"].tolist()
        bear_strikes = filtered_df[filtered_df["Trend"] == "Bear"]["Strike"].tolist()

        def find_nearest_strike(strikes, target):
            if not strikes:
                return None
            return min(strikes, key=lambda x: abs(x - target))

        nearest_bull_strike = find_nearest_strike(bull_strikes, atm_price)
        nearest_bear_strike = find_nearest_strike(bear_strikes, atm_price)

        # Display results in Streamlit
        with col6:
            if nearest_bull_strike is not None:
                st.markdown(f"- Nearest Bull Strike: `{nearest_bull_strike}`")
            else:
                st.markdown("- Nearest Bull Strike: None found")

            if nearest_bear_strike is not None:
                st.markdown(f"- Nearest Bear Strike: `{nearest_bear_strike}`")
            else:
                st.markdown("- Nearest Bear Strike: None found")

        st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")

