
# 📋 LunarSense-3 DATA ANNOTATION GUIDE

## Purpose
This guide provides instructions for human annotators to validate and label
lunar anomaly detection events from Chandrayaan-3 sensor data.

## Event Types

### THERMAL ANOMALIES (ChaSTE)
**Definition:** Unexpected temperature deviations indicating:
- Subsurface geothermal activity
- Impact events or disturbances
- Instrument malfunction (QC flag)

**Criteria for POSITIVE label:**
✅ Temperature deviation > 2 standard deviations from baseline
✅ Rapid thermal change (> 5 K/hour drift)
✅ Localized hot spot (max-min > 30 K range)
✅ Temporal clustering (multiple events in 6-hour window)

**Criteria for NEGATIVE label:**
❌ Normal diurnal cycle
❌ QC flag indicates instrument issue
❌ Gradual change < 1 K/hour

### SEISMIC EVENTS (ILSA)
**Definition:** Ground motion indicating:
- Moonquakes or seismic activity
- Impact-induced vibrations
- Marsquakes from nearby impacts

**Criteria for POSITIVE label:**
✅ STA/LTA ratio > 3.0 (significant acceleration)
✅ Amplitude > 0.1 m/s (measurable motion)
✅ RMS velocity > 0.05 m/s
✅ Temporal cluster (multiple events < 1 hour apart)

**Criteria for NEGATIVE label:**
❌ Low amplitude noise (< 0.01 m/s)
❌ QC flag indicates data quality issue
❌ Ambient seismic noise background

## Annotation Process

### Step 1: Review Data
1. Open event catalog CSV
2. Read event_id, timestamp, confidence
3. Check QC flags (0 = OK, 1,2,3 = FLAG)

### Step 2: Examine Signals
1. Plot thermal time series for ± 6 hours
2. Plot seismic waveform for ± 30 minutes
3. Compare with historical baseline

### Step 3: Label Event
Mark as: ANOMALY / NORMAL / UNCERTAIN

### Step 4: Add Comments
- Reason for classification
- Confidence level (1-5 scale)
- Data quality issues noted

## Quality Metrics

**Target Agreement:**
- Inter-annotator agreement: > 85%
- Model-human agreement: > 70%
- Consensus cases: > 90% confidence

**Common Errors to Avoid:**
❌ Labeling instrument artifacts as events
❌ Labeling normal background noise as anomalies
❌ Missing weak but real events (< 2σ threshold)
❌ Over-interpreting QC flags without data inspection

## Training Dataset

100 labeled samples provided:
- 40 confirmed ANOMALIES
- 40 confirmed NORMAL
- 20 UNCERTAIN (for discussion)

## Submission

1. Complete labels in provided CSV template
2. Include per-sample confidence (1-5)
3. Add overall quality assessment
4. Submit to: annotation@lunarsense.org

## Timeline

- Training: 2 hours
- Annotation: 50 samples/day
- Quality review: 24 hours
- Payment: $0.50/sample

## Questions?

Contact: annotations-support@lunarsense.org
