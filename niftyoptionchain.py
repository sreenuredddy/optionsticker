# --- Helper to get top N with positions ---
def get_top_strike_positions(df, col, n=3):
    ranked = df.nlargest(n, col)[["Strike", col]].reset_index(drop=True)
    ranked.index = ranked.index + 1  # Positions start at 1
    return ranked

# Current rankings
rank_ce_oi = get_top_strike_positions(filtered_df, "CE OI")
rank_pe_oi = get_top_strike_positions(filtered_df, "PE OI")
rank_ce_coi = get_top_strike_positions(filtered_df, "CE OI Chg")
rank_pe_coi = get_top_strike_positions(filtered_df, "PE OI Chg")

# Save previous rankings
prev_rank_ce_oi = st.session_state.get("prev_rank_ce_oi")
prev_rank_pe_oi = st.session_state.get("prev_rank_pe_oi")
prev_rank_ce_coi = st.session_state.get("prev_rank_ce_coi")
prev_rank_pe_coi = st.session_state.get("prev_rank_pe_coi")

def compare_ranks(prev_rank_df, curr_rank_df):
    shifts = []
    if prev_rank_df is None:
        return shifts
    for pos in curr_rank_df.index:
        strike_now = curr_rank_df.loc[pos, "Strike"]
        strike_prev = prev_rank_df.loc[pos, "Strike"] if pos in prev_rank_df.index else None
        if strike_now != strike_prev:
            shifts.append(f"🏆 Pos {pos}: {strike_prev} → {strike_now}")
    return shifts

# Compare for position changes
pos_shifts = {
    "CE OI": compare_ranks(prev_rank_ce_oi, rank_ce_oi),
    "PE OI": compare_ranks(prev_rank_pe_oi, rank_pe_oi),
    "CE OI Chg": compare_ranks(prev_rank_ce_coi, rank_ce_coi),
    "PE OI Chg": compare_ranks(prev_rank_pe_coi, rank_pe_coi)
}

# Save current ranks for next run
st.session_state["prev_rank_ce_oi"] = rank_ce_oi
st.session_state["prev_rank_pe_oi"] = rank_pe_oi
st.session_state["prev_rank_ce_coi"] = rank_ce_coi
st.session_state["prev_rank_pe_coi"] = rank_pe_coi

# --- Display in four columns ---
col1, col2, col3, col4 = st.columns(4)
for col, label, df_rank in zip(
    [col1, col2, col3, col4],
    ["CE OI", "PE OI", "CE OI Chg", "PE OI Chg"],
    [rank_ce_oi, rank_pe_oi, rank_ce_coi, rank_pe_coi]
):
    with col:
        st.markdown(f"### {label} Ranking")
        st.table(df_rank)  # Shows position, strike, value
        if label in pos_shifts and pos_shifts[label]:
            st.markdown("**Position Changes:**")
            for change in pos_shifts[label]:
                st.markdown(f"- {change}")
        else:
            st.markdown("No position changes.")

# Highlight #1 OI and #1 COI separately
st.markdown(f"🏅 **Highest CE OI**: Strike {rank_ce_oi.iloc[0]['Strike']} ({rank_ce_oi.iloc[0]['CE OI']:,.0f})")
st.markdown(f"🏅 **Highest PE OI**: Strike {rank_pe_oi.iloc[0]['Strike']} ({rank_pe_oi.iloc[0]['PE OI']:,.0f})")
st.markdown(f"🔥 **Highest CE OI Change**: Strike {rank_ce_coi.iloc[0]['Strike']} ({rank_ce_coi.iloc[0]['CE OI Chg']:,.0f})")
st.markdown(f"🔥 **Highest PE OI Change**: Strike {rank_pe_coi.iloc[0]['Strike']} ({rank_pe_coi.iloc[0]['PE OI Chg']:,.0f})")
