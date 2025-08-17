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

col1, col2, col3, col4 = st.columns(4)

with col1:
    # User input for expiry date
    expiry_input = st.text_input("📅 Enter Expiry Date (YYYY-MM-DD)", value="2025-08-21")
    # Choose index
    index = col2.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY", "SENSEX"],
        index=0,
        help="Select an index or type a custom one",
        accept_new_options=True
    )


# Auto-refresh logic (only if toggle is ON)
auto_refresh = col3.toggle("Auto Refresh (1 min)", value=True)

if auto_refresh:
    st_autorefresh(interval=60 * 1000, key="auto_refresh")
    col2.markdown(f"🔄 **Last Refreshed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col4:
    # Always show the manual refresh button on a new row
    if st.button("🔁 Manual Refresh"):
        st.session_state.last_refresh = time.time()
        st.rerun()

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
        num_strikes = st.slider("Number of strikes above/below", min_value=1, max_value=20, value=5)
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            
            st.markdown(f"""
            - 🟢 LTP: `{nifty_ltp:.2f}`
            - 🔄 Change: `{nifty_chg:.2f}` ({nifty_chg_pct:.2f}%)
            - ⚡ PCR: `{nifty_pcr}`
            """)

        with col6:
            st.markdown(f"""
            - 🎯 Previous Close: `{prev_close:.2f}`
            - 🧲 Closest Strike: `{nearest_strike}`
            - ⚡ Max Pain: `{max_pain_strike}`
            """)

        # --- Ensure numeric conversion for OI, OI Change, and Price Change columns ---
        user_strike = nearest_strike
        center_index = available_strikes.index(user_strike)
        start = max(center_index - num_strikes, 0)
        end = min(center_index + num_strikes + 1, len(available_strikes))
        selected_strikes = available_strikes[start:end]
        filtered_df = df[df["Strike"].isin(selected_strikes)].copy()
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

        # ✅ MODIFIED update_shift function to store old + new strike details, values, price changes
        def update_shift(key, old_strikes, new_strikes, new_val, new_price, old_val=None, old_price=None):
            old_strikes_str = ", ".join(map(str, old_strikes)) if old_strikes else "-"
            new_strikes_str = ", ".join(map(str, new_strikes)) if new_strikes else "-"
            if old_strikes != new_strikes:
                st.session_state["shift_tracker"][key] = (
                    f"{old_strikes_str} (Val: {old_val if old_val is not None else 0:,.0f}, "
                    f"Price: {old_price if old_price is not None else 0:+.2f}%) "
                    f"→ {new_strikes_str} (Val: {new_val:,.0f}, Price: {new_price:+.2f}%)"
                )
            elif key not in st.session_state["shift_tracker"]:
                st.session_state["shift_tracker"][key] = (
                    f"{new_strikes_str} (Val: {new_val:,.0f}, Price: {new_price:+.2f}%)"
                )

        # ===============================
        # 📌 Update tracker for all 8 metrics
        # ===============================
        update_shift("🚨 OI Call", st.session_state.get("last_oi_call"), max_ce_oi_strikes, max_ce_oi, max_ce_oi_price_chg,
                    st.session_state.get("last_oi_call_val"), st.session_state.get("last_oi_call_price"))

        update_shift("🚨 OI Put", st.session_state.get("last_oi_put"), max_pe_oi_strikes, max_pe_oi, max_pe_oi_price_chg,
                    st.session_state.get("last_oi_put_val"), st.session_state.get("last_oi_put_price"))

        update_shift("Abs OI Change Call", st.session_state.get("last_abs_call"), max_ce_oi_chg_abs_strikes, actual_ce_oi_chg_abs_val, max_ce_oi_chg_abs_price,
                    st.session_state.get("last_abs_call_val"), st.session_state.get("last_abs_call_price"))

        update_shift("Abs OI Change Put", st.session_state.get("last_abs_put"), max_pe_oi_chg_abs_strikes, actual_pe_oi_chg_abs_val, max_pe_oi_chg_abs_price,
                    st.session_state.get("last_abs_put_val"), st.session_state.get("last_abs_put_price"))

        update_shift("+OI Change Call", st.session_state.get("last_pos_call"), max_ce_oi_chg_pos_strikes, max_ce_oi_chg_pos, max_ce_oi_chg_pos_price,
                    st.session_state.get("last_pos_call_val"), st.session_state.get("last_pos_call_price"))

        update_shift("+OI Change Put", st.session_state.get("last_pos_put"), max_pe_oi_chg_pos_strikes, max_pe_oi_chg_pos, max_pe_oi_chg_pos_price,
                    st.session_state.get("last_pos_put_val"), st.session_state.get("last_pos_put_price"))

        update_shift("-OI Change Call", st.session_state.get("last_neg_call"), max_ce_oi_chg_neg_strikes, max_ce_oi_chg_neg, max_ce_oi_chg_neg_price,
                    st.session_state.get("last_neg_call_val"), st.session_state.get("last_neg_call_price"))

        update_shift("-OI Change Put", st.session_state.get("last_neg_put"), max_pe_oi_chg_neg_strikes, max_pe_oi_chg_neg, max_pe_oi_chg_neg_price,
                    st.session_state.get("last_neg_put_val"), st.session_state.get("last_neg_put_price"))

        # Save current values for next run comparisons
        st.session_state["last_oi_call"], st.session_state["last_oi_call_val"], st.session_state["last_oi_call_price"] = max_ce_oi_strikes, max_ce_oi, max_ce_oi_price_chg
        st.session_state["last_oi_put"], st.session_state["last_oi_put_val"], st.session_state["last_oi_put_price"] = max_pe_oi_strikes, max_pe_oi, max_pe_oi_price_chg
        st.session_state["last_abs_call"], st.session_state["last_abs_call_val"], st.session_state["last_abs_call_price"] = max_ce_oi_chg_abs_strikes, actual_ce_oi_chg_abs_val, max_ce_oi_chg_abs_price
        st.session_state["last_abs_put"], st.session_state["last_abs_put_val"], st.session_state["last_abs_put_price"] = max_pe_oi_chg_abs_strikes, actual_pe_oi_chg_abs_val, max_pe_oi_chg_abs_price
        st.session_state["last_pos_call"], st.session_state["last_pos_call_val"], st.session_state["last_pos_call_price"] = max_ce_oi_chg_pos_strikes, max_ce_oi_chg_pos, max_ce_oi_chg_pos_price
        st.session_state["last_pos_put"], st.session_state["last_pos_put_val"], st.session_state["last_pos_put_price"] = max_pe_oi_chg_pos_strikes, max_pe_oi_chg_pos, max_pe_oi_chg_pos_price
        st.session_state["last_neg_call"], st.session_state["last_neg_call_val"], st.session_state["last_neg_call_price"] = max_ce_oi_chg_neg_strikes, max_ce_oi_chg_neg, max_ce_oi_chg_neg_price
        st.session_state["last_neg_put"], st.session_state["last_neg_put_val"], st.session_state["last_neg_put_price"] = max_pe_oi_chg_neg_strikes, max_pe_oi_chg_neg, max_pe_oi_chg_neg_price

        # ===============================
        # 📌 Display Shift Updates in Two Columns
        # ===============================
        st.markdown("### 📢 OI & OI ChgShift Updates ")

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
        with col7:
            # --- Directional recommendation ---
            st.markdown(f"""
            - Top Skew Strike:`{top_strike}`
            - vtySkw:`{top_skew:.2f}`
            - Dominant Driver:`{dominant}`
            """)


        st.plotly_chart(fig, use_container_width=True)
        # --- Display table ---
        #st.dataframe(df.sort_values("vtySkw", ascending=False), use_container_width=True)

        # --- Convert CE and PE volumes to numeric for proper comparison ---
        filtered_df["CE Volume"] = pd.to_numeric(filtered_df["CE Volume"], errors="coerce")
        filtered_df["PE Volume"] = pd.to_numeric(filtered_df["PE Volume"], errors="coerce")

        # --- Get top 3 CE ---
        top3_ce_vol_strikes = (
            filtered_df.nlargest(3, "CE Volume")[["Strike", "CE Volume", "CE prcOIA", "CE LTP", "CE (OI/Chg)"]]
            .rename(columns={"CE prcOIA": "CE Interpretation"})
        )
        top3_ce_vol_strikes.index = range(1, len(top3_ce_vol_strikes) + 1)

        # --- Get top 3 PE ---
        top3_pe_vol_strikes = (
            filtered_df.nlargest(3, "PE Volume")[["Strike", "PE Volume", "PE prcOIA", "PE LTP", "PE (OI/Chg)"]]
            .rename(columns={"PE prcOIA": "PE Interpretation"})
        )
        top3_pe_vol_strikes.index = range(1, len(top3_pe_vol_strikes) + 1)

        # --- Detect interpretation for CE ---
        ce_msg, ce_status = "", ""
        if len(top3_ce_vol_strikes) >= 2:
            ce_highest = top3_ce_vol_strikes.iloc[0]["Strike"]
            ce_second  = top3_ce_vol_strikes.iloc[1]["Strike"]
            if ce_second < ce_highest:
                ce_msg = f"🔴 **Resistance Strong** – forming at **{int(ce_second)}** (below highest {int(ce_highest)})"
                ce_status = "Resistance Strong"
            elif ce_second > ce_highest:
                ce_msg = f"🟠 **Resistance Weak** – shifting to **{int(ce_second)}** (above highest {int(ce_highest)})"
                ce_status = "Resistance Weak"

        # --- Detect interpretation for PE ---
        pe_msg, pe_status = "", ""
        if len(top3_pe_vol_strikes) >= 2:
            pe_highest = top3_pe_vol_strikes.iloc[0]["Strike"]
            pe_second  = top3_pe_vol_strikes.iloc[1]["Strike"]
            if pe_second > pe_highest:
                pe_msg = f"🟢 **Support Strong** – forming at **{int(pe_second)}** (above highest {int(pe_highest)})"
                pe_status = "Support Strong"
            elif pe_second < pe_highest:
                pe_msg = f"🟠 **Support Weak** – shifting to **{int(pe_second)}** (below highest {int(pe_highest)})"
                pe_status = "Support Weak"

        # --- Final Market View ---
        market_msg = ""
        if ce_status and pe_status:
            if ce_status == "Resistance Strong" and pe_status == "Support Strong":
                market_msg = "📊 **Market Sideways** – Both Support & Resistance are Strong"
            elif ce_status == "Resistance Strong" and pe_status == "Support Weak":
                market_msg = "🔻 **Market Bearish** – Strong Resistance, Weak Support"
            elif ce_status == "Resistance Weak" and pe_status == "Support Strong":
                market_msg = "🔺 **Market Bullish** – Strong Support, Weak Resistance"

        # --- Display results in two columns with markdown ---
        st.markdown("### 📊 Top 3 Highest Volume Strikes")
        col1, col2 = st.columns(2)

        # --- Show Final Market View ---
        if market_msg:
            #col4.markdown("---")
            col8.markdown(f" 📌 Final Market View: {market_msg}")

        with col1:
            st.markdown("#### 🟢 Call Options (CE)")
            st.dataframe(top3_ce_vol_strikes.style.hide(axis="index"), use_container_width=True)
        if ce_msg:
            col8.markdown(ce_msg)

        with col2:
            st.markdown("#### 🔴 Put Options (PE)")
            st.dataframe(top3_pe_vol_strikes.style.hide(axis="index"), use_container_width=True)
        if pe_msg:
            col8.markdown(pe_msg)


        #CE AND PE comparison

        # Define filters for each table
        # 🎯 Define filters
        filter_1 = (filtered_df["CE prcOIA"] == "Long Bldp") & (filtered_df["PE prcOIA"] == "Short Bldp")
        filter_2 = (filtered_df["CE prcOIA"] == "Short Covering") & (filtered_df["PE prcOIA"] == "Short Bldp")
        filter_3 = (filtered_df["CE prcOIA"] == "Short Bldp") & (filtered_df["PE prcOIA"] == "Long Bldp")
        filter_4 = (filtered_df["CE prcOIA"] == "Short Bldp") & (filtered_df["PE prcOIA"] == "Short Covering")

        # Get strikes as comma-separated strings
        df1_strikes = ", ".join(map(str, filtered_df.loc[filter_1, "Strike"].tolist()))
        df2_strikes = ", ".join(map(str, filtered_df.loc[filter_2, "Strike"].tolist()))
        df3_strikes = ", ".join(map(str, filtered_df.loc[filter_3, "Strike"].tolist()))
        df4_strikes = ", ".join(map(str, filtered_df.loc[filter_4, "Strike"].tolist()))

        # 📌 Create 4 columns for the 4 conditions
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("CE: Long Bldp <br> PE: Short Bldp", unsafe_allow_html=True)
            st.write(df1_strikes if df1_strikes else "—")

        with col2:
            st.markdown("CE: Short Covering <br> PE: Short Bldp", unsafe_allow_html=True)
            st.write(df2_strikes if df2_strikes else "—")

        with col3:
            st.markdown("CE: Short Bldp <br> PE: Long Bldp", unsafe_allow_html=True)
            st.write(df3_strikes if df3_strikes else "—")

        with col4:
            st.markdown("CE: Short Bldp <br> PE: Short Covering", unsafe_allow_html=True)
            st.write(df4_strikes if df4_strikes else "—")

        # =======================
        # --- Ensure numeric conversion for OI, OI Change, and Price Change columns ---
        cols = ["CE OI", "PE OI", "CE OI Chg", "PE OI Chg", "CE Chg%", "PE Chg%"]
        for col in cols:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)

        # =======================

        # --- Helper to get top N strikes for a column ---
        def get_top_strikes(df, col, n=3):
            return df.nlargest(n, col)["Strike"].tolist()

        # --- Get Current Top 3 Lists ---
        top3_ce_oi = get_top_strikes(filtered_df, "CE OI")
        top3_pe_oi = get_top_strikes(filtered_df, "PE OI")
        top3_ce_oi_chg = get_top_strikes(filtered_df, "CE OI Chg")
        top3_pe_oi_chg = get_top_strikes(filtered_df, "PE OI Chg")

        # --- Previous data from session_state or defaults ---
        prev_top3_ce_oi = st.session_state.get("prev_top3_ce_oi", [])
        prev_top3_pe_oi = st.session_state.get("prev_top3_pe_oi", [])
        prev_top3_ce_oi_chg = st.session_state.get("prev_top3_ce_oi_chg", [])
        prev_top3_pe_oi_chg = st.session_state.get("prev_top3_pe_oi_chg", [])

        # --- Function to find entries/exits in Top 3 ---
        def find_shift(prev_list, curr_list):
            entered = [s for s in curr_list if s not in prev_list]
            exited = [s for s in prev_list if s not in curr_list]
            return entered, exited

        # --- Track shift details for all four metrics ---
        shifts_top3 = {}

        for label, prev, curr in [
            ("CE OI", prev_top3_ce_oi, top3_ce_oi),
            ("PE OI", prev_top3_pe_oi, top3_pe_oi),
            ("CE OI Chg", prev_top3_ce_oi_chg, top3_ce_oi_chg),
            ("PE OI Chg", prev_top3_pe_oi_chg, top3_pe_oi_chg),
        ]:
            entered, exited = find_shift(prev, curr)
            shifts_top3[label] = {"entered": entered, "exited": exited}

        # --- Save current as prev for next refresh ---
        st.session_state["prev_top3_ce_oi"] = top3_ce_oi
        st.session_state["prev_top3_pe_oi"] = top3_pe_oi
        st.session_state["prev_top3_ce_oi_chg"] = top3_ce_oi_chg
        st.session_state["prev_top3_pe_oi_chg"] = top3_pe_oi_chg

        # --- Display in four columns ---
        col1, col2, col3, col4 = st.columns(4)

        for col, label in zip([col1, col2, col3, col4], shifts_top3.keys()):
            with col:
                st.markdown(f"### {label} Top 3 Shift")
                if shifts_top3[label]["entered"]:
                    st.markdown(f"🆕 Entered: {', '.join(map(str, shifts_top3[label]['entered']))}")
                else:
                    st.markdown("🆕 Entered: —")
                if shifts_top3[label]["exited"]:
                    st.markdown(f"🔻 Exited: {', '.join(map(str, shifts_top3[label]['exited']))}")
                else:
                    st.markdown("🔻 Exited: —")

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
        show_new_money_flow_overall_split(filtered_df, top_n=num_strikes)

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


        # --- Ensure numeric conversion ---
        num_cols = ["CE OI", "CE OI Chg", "CE Volume", "PE OI", "PE OI Chg", "PE Volume"]
        filtered_df[num_cols] = filtered_df[num_cols].apply(pd.to_numeric, errors="coerce")

        # --- Find maxima for all required columns ---
        max_vals = {
            "CE OI":      filtered_df["CE OI"].max(),
            "CE OI Chg":  filtered_df["CE OI Chg"].max(),
            "CE Volume":  filtered_df["CE Volume"].max(),
            "PE OI":      filtered_df["PE OI"].max(),
            "PE OI Chg":  filtered_df["PE OI Chg"].max(),
            "PE Volume":  filtered_df["PE Volume"].max(),
        }

        # --- Identify Strong Resistance and Support strikes ---
        filtered_df["Strong_CE"] = (
            (filtered_df["CE OI"] == max_vals["CE OI"]) &
            (filtered_df["CE OI Chg"] == max_vals["CE OI Chg"]) &
            (filtered_df["CE Volume"] == max_vals["CE Volume"])
        )

        filtered_df["Strong_PE"] = (
            (filtered_df["PE OI"] == max_vals["PE OI"]) &
            (filtered_df["PE OI Chg"] == max_vals["PE OI Chg"]) &
            (filtered_df["PE Volume"] == max_vals["PE Volume"])
        )

        # --- Extract strikes ---
        resist_strikes  = filtered_df.loc[filtered_df["Strong_CE"], "Strike"].astype(str).tolist()
        support_strikes = filtered_df.loc[filtered_df["Strong_PE"], "Strike"].astype(str).tolist()

        with col7:
            # --- Markdown Output ---
            st.markdown(
                f"🔴 **Strong Resistance** at strike(s): `{', '.join(resist_strikes)}`"
                if resist_strikes else "### 🔴 **Strong Resistance**: _None detected_"
            )

            st.markdown(
                f"🟢 **Strong Support** at strike(s): `{', '.join(support_strikes)}`"
                if support_strikes else "### 🟢 **Strong Support**: _None detected_"
            )

        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
        from datetime import datetime

        def atm_vega_theta_monitor(filtered_df, nifty_ltp):
            atm_strike = min(filtered_df["Strike"], key=lambda x: abs(x - nifty_ltp))
            atm_row = filtered_df[filtered_df["Strike"] == atm_strike].iloc[0]
            ce_vega = float(atm_row.get("CE Vega", 0))
            pe_vega = float(atm_row.get("PE Vega", 0))
            ce_theta = float(atm_row.get("CE Theta", 0))
            pe_theta = float(atm_row.get("PE Theta", 0))
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_row = {
                "Time": now,
                "Strike": atm_strike,
                "CE Vega": ce_vega,
                "PE Vega": pe_vega,
                "CE Theta": ce_theta,
                "PE Theta": pe_theta,
            }

            # Session state for persistent storage
            if "atm_vt_log" not in st.session_state:
                st.session_state["atm_vt_log"] = []
            st.session_state["atm_vt_log"].append(current_row)
            st.session_state["atm_vt_log"] = st.session_state["atm_vt_log"][-500:]

            df = pd.DataFrame(st.session_state["atm_vt_log"])

            # Compute differences
            for col in ["CE Vega", "PE Vega", "CE Theta", "PE Theta"]:
                df["Δ " + col] = df[col].diff()

            # Spike logic with thresholds
            for diff_col in ["Δ CE Vega", "Δ PE Vega", "Δ CE Theta", "Δ PE Theta"]:
                # Guards for few points
                if diff_col not in df or df[diff_col].isnull().all():
                    df["Spike " + diff_col] = False
                else:
                    roll_mean = df[diff_col].abs().rolling(window=10, min_periods=2).mean()
                    roll_std = df[diff_col].abs().rolling(window=10, min_periods=2).std()
                    threshold = roll_mean + 1 * roll_std
                    df["Spike " + diff_col] = (df[diff_col].abs() > threshold).fillna(False)

            # Always add/refresh the Spike Direction column, safe for first rows!
            def calc_direction(row):
                msg = ""
                if row.get("Spike Δ CE Vega", False):
                    if pd.notnull(row.get("Δ CE Vega")):
                        if row["Δ CE Vega"] > 0:
                            msg += "🔼 CE Vega ↑ (Bullish); "
                        else:
                            msg += "🔽 CE Vega ↓ (Bearish); "
                if row.get("Spike Δ PE Vega", False):
                    if pd.notnull(row.get("Δ PE Vega")):
                        if row["Δ PE Vega"] > 0:
                            msg += "🔽 PE Vega ↑ (Bearish); "
                        else:
                            msg += "🔼 PE Vega ↓ (Bullish); "
                if row.get("Spike Δ CE Theta", False):
                    if pd.notnull(row.get("Δ CE Theta")):
                        if row["Δ CE Theta"] > 0:
                            msg += "⚡ CE Theta ↑ (Bearish/time decay); "
                        else:
                            msg += "⭐ CE Theta ↓ (Bullish); "
                if row.get("Spike Δ PE Theta", False):
                    if pd.notnull(row.get("Δ PE Theta")):
                        if row["Δ PE Theta"] > 0:
                            msg += "⚡ PE Theta ↑ (Bullish/time decay); "
                        else:
                            msg += "⭐ PE Theta ↓ (Bearish); "
                return msg.strip()

            df["Spike Direction"] = df.apply(calc_direction, axis=1)

            st.markdown("### 🟩 ATM Vega/Theta Differences, Spike & Direction")
            st.dataframe(
                df[::-1][[
                    "Time","Strike",
                    "CE Vega","PE Vega","CE Theta","PE Theta",
                    "Δ CE Vega","Δ PE Vega","Δ CE Theta","Δ PE Theta",
                    "Spike Δ CE Vega","Spike Δ PE Vega","Spike Δ CE Theta","Spike Δ PE Theta",
                    "Spike Direction"
                ]],
                use_container_width=True,
                height=420
            )

            # Vega Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Time"], y=df["Δ CE Vega"], mode="lines+markers", name="Δ CE Vega"
            ))
            fig.add_trace(go.Scatter(
                x=df["Time"], y=df["Δ PE Vega"], mode="lines+markers", name="Δ PE Vega"
            ))
            fig.update_layout(title="ATM Vega Differences Over Time", xaxis_title="Time", yaxis_title="Vega Diff")
            st.plotly_chart(fig, use_container_width=True)

            # Theta Plot
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df["Time"], y=df["Δ CE Theta"], mode="lines+markers", name="Δ CE Theta"
            ))
            fig2.add_trace(go.Scatter(
                x=df["Time"], y=df["Δ PE Theta"], mode="lines+markers", name="Δ PE Theta"
            ))
            fig2.update_layout(title="ATM Theta Differences Over Time", xaxis_title="Time", yaxis_title="Theta Diff")
            st.plotly_chart(fig2, use_container_width=True)

        # USAGE:
        # atm_vega_theta_monitor(filtered_df, nifty_ltp)

        atm_vega_theta_monitor(filtered_df, top_strike)

    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")

