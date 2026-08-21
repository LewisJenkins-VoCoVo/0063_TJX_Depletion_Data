# -------------------------------------------------
# IDENTIFY BATTERY-POWERED DEVICES
# -------------------------------------------------
battery_rows = []

for file in sorted(IPEI_DIR.glob("*/*.csv")):
    ipei = file.stem.strip()

    trace = pd.read_csv(file)

    if "snapshot_time" not in trace.columns or "battery_level" not in trace.columns:
        continue

    trace["battery_level"] = pd.to_numeric(trace["battery_level"], errors="coerce")
    trace = trace.dropna(subset=["battery_level"])

    if trace.empty:
        continue

    sample_count = len(trace)
    min_soc = trace["battery_level"].min()
    max_soc = trace["battery_level"].max()
    soc_range = max_soc - min_soc

    always_high = min_soc >= HIGH_STATIC_MIN and max_soc <= 100
    always_low = min_soc >= 0 and max_soc <= LOW_STATIC_MAX
    static_extreme = always_high or always_low

    battery_powered = (
        sample_count >= MIN_VALID_SAMPLES and
        soc_range >= MIN_SOC_VARIATION and
        not static_extreme
    )

    battery_rows.append({
        "ipei": ipei,
        "battery_powered": battery_powered,
        "sample_count": sample_count,
        "min_soc": min_soc,
        "max_soc": max_soc,
        "soc_range": soc_range,
    })

battery_df = pd.DataFrame(battery_rows)

battery_powered_ipeis = set(
    battery_df.loc[battery_df["battery_powered"], "ipei"]
)

df = df[df["ipei"].isin(battery_powered_ipeis)].copy()