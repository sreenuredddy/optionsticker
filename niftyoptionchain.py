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
st.set_page_config(page_title="NIFTY Option Chain - Calls & Puts", layout="wide")
st.title("📈 NIFTY Option Chain (Calls vs Puts)")
REFRESH_INTERVAL = 60  # seconds

col1, col2, col5 = st.columns(3)

with col1:
    # User input for expiry date
    expiry_input = st.text_input("📅 Enter Expiry Date (YYYY-MM-DD)", value="2025-08-14")

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
    API_URL = f"https://nw.nuvamawealth.com/edelmw-content/content/new-options/option-chain/NFO/OPTIDX/NIFTY/{expiry_input}?screen=all"
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

        col3, col4 = st.columns(2)

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

        # --- Directional recommendation ---
        st.markdown("### 🔎 Top Skew Strike Analysis")
        st.markdown(f"- **Top Skew Strike:** `{top_strike}`")
        st.markdown(f"- **vtySkw:** `{top_skew:.2f}`")
        st.markdown(f"- **Dominant Driver:** `{dominant}`")

        st.plotly_chart(fig, use_container_width=True)
        # --- Display table ---
        st.dataframe(df.sort_values("vtySkw", ascending=False), use_container_width=True)



        #ce and pe buildup
        # Function to filter filtered_df by prcOIA category for a given option side ("CE" or "PE")
        def filter_by_prcoia(df, side, category):
            prcOIA_col = f"{side} prcOIA"
            return df[df[prcOIA_col] == category][["Strike", f"{side} (OI/Chg)", f"{side} Chg%", prcOIA_col]]

        # The four prcOIA classification categories to display
        prcoia_categories = ["Long Bldp", "Short Covering", "Short Bldp", "Long Unwndg"]

        st.markdown("### ⚡ Classified Strike Categories Using prcOIA")

        # Display Call Option (CE) classification tables side by side
        st.markdown("#### Call Options (CE) Analysis")
        ce_cols = st.columns(2)
        for i, category in enumerate(prcoia_categories):
            with ce_cols[i % 2]:
                df_cat = filter_by_prcoia(filtered_df, "CE", category)
                st.markdown(f"**{category}**")
                if df_cat.empty:
                    st.markdown("_No strikes found._")
                else:
                    st.dataframe(df_cat, use_container_width=True, height=200)

        # Display Put Option (PE) classification tables side by side
        st.markdown("#### Put Options (PE) Analysis")
        pe_cols = st.columns(2)
        for i, category in enumerate(prcoia_categories):
            with pe_cols[i % 2]:
                df_cat = filter_by_prcoia(filtered_df, "PE", category)
                st.markdown(f"**{category}**")
                if df_cat.empty:
                    st.markdown("_No strikes found._")
                else:
                    st.dataframe(df_cat, use_container_width=True, height=200)

        #volume
        # After you have 'filtered_df' DataFrame ready from your existing code

        # Convert CE and PE volumes to numeric for proper comparison
        filtered_df["CE Volume"] = pd.to_numeric(filtered_df["CE Volume"], errors="coerce")
        filtered_df["PE Volume"] = pd.to_numeric(filtered_df["PE Volume"], errors="coerce")

        # Find maximum volumes
        max_ce_volume = filtered_df["CE Volume"].max()
        max_pe_volume = filtered_df["PE Volume"].max()

        # Get rows with the maximum CE volume
        max_ce_vol_strikes = filtered_df[filtered_df["CE Volume"] == max_ce_volume][["Strike", "CE Volume", "CE prcOIA", "CE LTP", "CE (OI/Chg)"]]

        # Get rows with the maximum PE volume
        max_pe_vol_strikes = filtered_df[filtered_df["PE Volume"] == max_pe_volume][["Strike", "PE Volume", "PE prcOIA", "PE LTP", "PE (OI/Chg)"]]

        # Display results
        st.markdown("## Highest Volume Strikes")

        st.markdown("### Call Options (CE) with Highest Volume")
        if not max_ce_vol_strikes.empty:
            st.dataframe(max_ce_vol_strikes)
        else:
            st.write("No data available for highest Call option volume strikes.")

        st.markdown("### Put Options (PE) with Highest Volume")
        if not max_pe_vol_strikes.empty:
            st.dataframe(max_pe_vol_strikes)
        else:
            st.write("No data available for highest Put option volume strikes.")

        #IV Bias and Trend

        # === Step 1: Convert necessary columns to numeric ===
        numeric_cols = [
            "CE OI Chg", "PE OI Chg", "CE Chg%", "PE Chg%",
            "CE IV Fut @ LTP", "PE IV Fut @ LTP", "Strike"
        ]
        for col in numeric_cols:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')

        # Drop NaNs in critical columns
        filtered_df = filtered_df.dropna(subset=numeric_cols)

        # === Step 2: Market Trend Inference ===
        def infer_market_trend(row):
            ce_oi, pe_oi = row["CE OI Chg"], row["PE OI Chg"]
            ce_price, pe_price = row["CE Chg%"], row["PE Chg%"]

            ce_oi_up, ce_oi_down = ce_oi > 0, ce_oi < 0
            pe_oi_up, pe_oi_down = pe_oi > 0, pe_oi < 0
            ce_price_up, ce_price_down = ce_price > 0, ce_price < 0
            pe_price_up, pe_price_down = pe_price > 0, pe_price < 0

            if ce_oi_up and pe_oi_up:
                if ce_price_up and pe_price_down:
                    return "Bullish : Call Long Bldp & Put Unwinding"
                elif ce_price_down and pe_price_up:
                    return "Bearish : Call Short Bldp & Put Long Bldp"
                elif ce_price_up and pe_price_up:
                    return "Indecision : (Calls and Puts Both Rising)"
                elif ce_price_down and pe_price_down:
                    return "Sideways : (Calls and Puts Both Falling)"
                else:
                    return "Mixed Bullish Signals"

            elif ce_oi_up and pe_oi_down:
                if ce_price_up and pe_price_down:
                    return "Bullish : Long Bldp Calls & Put Unwinding"
                elif ce_price_down and pe_price_up:
                    return "Bearish : Short Bldp Calls & Put Long Bldp"
                else:
                    return "Calls Dominant, Puts Unwinding"

            elif ce_oi_down and pe_oi_up:
                if ce_price_down and pe_price_up:
                    return "Bearish Puts Buying & Call Selling"
                elif ce_price_up and pe_price_down:
                    return "Bullish : Short Bldp Puts & Bullish : Short Covering Calls"
                else:
                    return "Puts Dominant, Calls Unwinding"

            elif ce_oi_down and pe_oi_down:
                if ce_price_up and pe_price_up:
                    return "Long Unwinding Both Calls and Puts"
                elif ce_price_down and pe_price_down:
                    return "Short Covering Both Calls and Puts"
                else:
                    return "Mixed OI Down and Price Signals"

            elif abs(ce_oi) < 1e-3 or abs(pe_oi) < 1e-3:
                return "Low OI Change - Neutral or Low Activity"

            elif abs(ce_oi) < 1e-3 and (ce_price_up or ce_price_down):
                return "Call Price Move without OI Change (Possible Speculative)"

            elif abs(pe_oi) < 1e-3 and (pe_price_up or pe_price_down):
                return "Put Price Move without OI Change (Possible Speculative)"

            else:
                return "Unclear / Mixed Signals"

        # === Step 3: Row-wise IV Skew with Mirrored Strike Matching ===
        def rowwise_iv_skew(df, atm_strike):
            iv_diffs = []
            compared_pairs = []

            for _, row in df.iterrows():
                strike = row["Strike"]
                diff = None
                pair = None

                mirror_put_strike = atm_strike + (atm_strike - strike)
                ce_iv = row["CE IV Fut @ LTP"]
                pe_iv_list = df.loc[df["Strike"] == mirror_put_strike, "PE IV Fut @ LTP"].values
                if len(pe_iv_list) > 0:
                    diff = ce_iv - pe_iv_list[0]
                    pair = f"{strike}C vs {mirror_put_strike}P"

                iv_diffs.append(diff if diff is not None else 0)
                compared_pairs.append(pair if pair is not None else "N/A")

            return pd.Series(iv_diffs), pd.Series(compared_pairs)

        # === Step 4: Find ATM Strike ===
        atm_strike = min(filtered_df["Strike"], key=lambda x: abs(x - nearest_strike))

        # === Step 5: Calculate IV Differences & Avg Skew ===
        filtered_df["IV Difference"], filtered_df["Compared Pair"] = rowwise_iv_skew(filtered_df, atm_strike)

        # === Step 6: Apply Market Trend ===
        filtered_df["Market Trend"] = filtered_df.apply(infer_market_trend, axis=1)
        filtered_df["IV Difference"] = filtered_df["IV Difference"].round(2)

        # === Step 7: Sum IV Differences Above & Below ATM ===
        above_atm_sum = filtered_df.loc[filtered_df["Strike"] > atm_strike, "IV Difference"].sum()
        below_atm_sum = filtered_df.loc[filtered_df["Strike"] < atm_strike, "IV Difference"].sum()

        if below_atm_sum > above_atm_sum:
            iv_bias_summary = "📈 Bullish Bias (Below ATM IV lower)"
        elif above_atm_sum < below_atm_sum:
            iv_bias_summary = "📉 Bearish Bias (Above ATM IV lower)"
        else:
            iv_bias_summary = "⚖️ Neutral Bias"

        # === Step 8: Display ===
        st.markdown("### 🔍 Market Trend + Equidistant IV Signals by Strike")
        st.dataframe(
            filtered_df[[
                "Strike", "Compared Pair", "CE OI Chg", "CE Chg%", "PE OI Chg", "PE Chg%",
                "CE IV Fut @ LTP", "PE IV Fut @ LTP", "IV Difference", "Market Trend"
            ]].sort_values("Strike"),
            use_container_width=True,
            height=420
        )

        st.markdown(f"**Below ATM IV Sum:** {below_atm_sum:.2f} | **Above ATM IV Sum:** {above_atm_sum:.2f}")
        st.markdown(f"**Overall IV Bias:** {iv_bias_summary}")


        #New Money Flow
        
        def show_new_money_flow(filtered_df, reference_price, label, strike_col="Strike"):
            # Ensure numeric columns
            filtered_df["CE OI Chg"] = pd.to_numeric(filtered_df["CE OI Chg"], errors="coerce")
            filtered_df["PE OI Chg"] = pd.to_numeric(filtered_df["PE OI Chg"], errors="coerce")

            # Find ATM/Ref strike based on reference price
            ref_strike = min(filtered_df[strike_col], key=lambda x: abs(x - reference_price))
            st.markdown(f"**{label} Strike:** `{ref_strike}`")

            # Get OI change at reference strike
            ref_row = filtered_df[filtered_df[strike_col] == ref_strike]
            ref_ce_oi_chg = ref_row["CE OI Chg"].values[0] if not ref_row.empty else 0
            ref_pe_oi_chg = ref_row["PE OI Chg"].values[0] if not ref_row.empty else 0

            # Filter for strikes with higher OI change than reference
            new_money_strikes = filtered_df[
                (filtered_df["CE OI Chg"].abs() > abs(ref_ce_oi_chg)) |
                (filtered_df["PE OI Chg"].abs() > abs(ref_pe_oi_chg))
            ].copy()

            # Add side marker
            def side_marker(row):
                ce = abs(row["CE OI Chg"]) > abs(ref_ce_oi_chg) and row["CE OI Chg"] > 0
                pe = abs(row["PE OI Chg"]) > abs(ref_pe_oi_chg) and row["PE OI Chg"] > 0
                if ce and pe:
                    return "Both"
                elif ce:
                    return "CE"
                elif pe:
                    return "PE"
                else:
                    return ""
            new_money_strikes["New Money Side"] = new_money_strikes.apply(side_marker, axis=1)

            # Sort
            new_money_strikes["Max OI Chg"] = new_money_strikes[["CE OI Chg", "PE OI Chg"]].abs().max(axis=1)
            new_money_strikes = new_money_strikes.sort_values("Max OI Chg", ascending=False)

            # === Summary Calculations ===
            total_ce_all = filtered_df["CE OI Chg"].sum()
            total_pe_all = filtered_df["PE OI Chg"].sum()

            total_ce_new_money = new_money_strikes["CE OI Chg"].sum()
            total_pe_new_money = new_money_strikes["PE OI Chg"].sum()

            # === Display ===
            st.markdown(f"### 📊 Strikes Where New Money Is Flowing (Higher OI Change than {label} Strike)")
            if new_money_strikes.empty:
                st.write(f"No strikes currently have OI change exceeding that at the {label.lower()} strike.")
            else:
                st.dataframe(
                    new_money_strikes[
                        ["Strike", "CE OI Chg", "CE Chg%", "PE OI Chg", "PE Chg%", "CE prcOIA", "PE prcOIA", "New Money Side"]
                    ],
                    use_container_width=True,
                    height=420
                )
            # Display summaries
            st.markdown(f"""
            **🔹 OI Change Summary (All Strikes)**  
            - Total CE OI Change: `{total_ce_all:.0f}`  
            - Total PE OI Change: `{total_pe_all:.0f}`  

            **🔹 OI Change Summary (New Money Strikes)**  
            - Total CE OI Change: `{total_ce_new_money:.0f}`  
            - Total PE OI Change: `{total_pe_new_money:.0f}`
            """)



        # --- Use the function for both Previous Close and Spot (ATM) reference strikes ---
        # Previous Close
        show_new_money_flow(filtered_df, prev_close, "Previous Close")

        # ATM (spot)
        show_new_money_flow(filtered_df, nifty_ltp, "ATM (by spot NIFTY)")

        # --- STRADDLE CALCULATIONS ---

        # Ensure numeric values for price columns
        filtered_df["CE Open"] = pd.to_numeric(filtered_df.get("CE Open", 0), errors="coerce")
        filtered_df["PE Open"] = pd.to_numeric(filtered_df.get("PE Open", 0), errors="coerce")
        filtered_df["CE Prev Close"] = pd.to_numeric(filtered_df.get("CE Prev Close", 0), errors="coerce")
        filtered_df["PE Prev Close"] = pd.to_numeric(filtered_df.get("PE Prev Close", 0), errors="coerce")
        filtered_df["CE LTP"] = pd.to_numeric(filtered_df.get("CE LTP", 0), errors="coerce")
        filtered_df["PE LTP"] = pd.to_numeric(filtered_df.get("PE LTP", 0), errors="coerce")

        # Calculate Straddle Prices
        filtered_df["Opening Straddle"] = filtered_df["CE Open"] + filtered_df["PE Open"]
        filtered_df["Previous Day-End Straddle"] = filtered_df["CE Prev Close"] + filtered_df["PE Prev Close"]
        filtered_df["Current Straddle"] = filtered_df["CE LTP"] + filtered_df["PE LTP"]

        # --- Find ATM Strike (by NIFTY spot price) ---
        atm_strike = min(filtered_df["Strike"], key=lambda x: abs(x - nifty_ltp))

        # Calculate previous straddle values for inference
        filtered_df["Prev Close Straddle"] = filtered_df["CE Prev Close"] + filtered_df["PE Prev Close"]

        # Function to infer straddle directional bias based on dominant leg and straddle movement
        def infer_straddle_direction(row, atm_strike):
            strike = row["Strike"]
            curr_straddle = row["Current Straddle"]
            prev_straddle = row["Prev Close Straddle"]
            open_straddle = row["Opening Straddle"]
            ce_ltp = row["CE LTP"]
            pe_ltp = row["PE LTP"]

            # Check for missing data
            if pd.isna(curr_straddle) or pd.isna(prev_straddle) or pd.isna(ce_ltp) or pd.isna(pe_ltp) or pd.isna(open_straddle):
                return ""

            straddle_change = curr_straddle - prev_straddle
            Straddletoday_change = curr_straddle - open_straddle

            # Determine dominant leg (higher premium leg)
            dominant_leg = "CE" if ce_ltp > pe_ltp else "PE"

            # Directional inference logic based on your explanation:
            if strike > atm_strike:
                # Dominant leg expected to be PE
                # If straddle is decreasing because put leg drops, bullish signal
                if straddle_change < 0 and Straddletoday_change < 0  and dominant_leg == "PE":
                    return "Short Straddle"
                elif straddle_change > 0 and Straddletoday_change > 0 and dominant_leg == "PE":
                    return "Long Straddle"
                else:
                    return "Neutral"
            elif strike < atm_strike:
                # Dominant leg expected to be CE
                # If straddle is decreasing because call leg drops, bearish signal
                if straddle_change < 0 and Straddletoday_change < 0 and dominant_leg == "CE":
                    return "Short Straddle"
                elif straddle_change > 0 and Straddletoday_change > 0 and dominant_leg == "CE":
                    return "Long Straddle"
                else:
                    return "Neutral"
            else:
                # At ATM strike, directional signal less clear
                return "Neutral"

        filtered_df["Straddle Direction"] = filtered_df.apply(lambda row: infer_straddle_direction(row, atm_strike), axis=1)

        # --- Display Straddle Table with Direction ---
        st.markdown("### 🧮 Straddle Prices and Directional Bias by Strike")
        st.dataframe(
            filtered_df[["Strike", "Previous Day-End Straddle", "Opening Straddle", "Current Straddle", "Straddle Direction"]],
            use_container_width=True,
            height = 420
        )

        # --- Optional: Plot Straddle and Directional Bias ---
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df["Strike"], y=filtered_df["Opening Straddle"],
            mode="lines+markers", name="Opening"
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df["Strike"], y=filtered_df["Previous Day-End Straddle"],
            mode="lines+markers", name="Day-End"
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df["Strike"], y=filtered_df["Current Straddle"],
            mode="lines+markers", name="Current"
        ))
        # Add directional bias as text annotations on current straddle line
        fig.add_trace(go.Scatter(
            x=filtered_df["Strike"], y=filtered_df["Current Straddle"],
            mode="text",
            text=filtered_df["Straddle Direction"],
            textposition="top center",
            showlegend=False
        ))

        fig.update_layout(
            title="Straddle Values and Directional Bias by Strike",
            xaxis_title="Strike Price",
            yaxis_title="Straddle Price"
        )
        st.plotly_chart(fig, use_container_width=True)


        st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")

