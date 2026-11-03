# Chapter 5, Figure 5.28

This code **replots published numerical measurements**; it does not execute DeepAnomaly, Faster R-CNN, FieldSAFE inference, or a new experiment. The distances refer to detection-performance ranges, not confidence intervals or physical maximum range.

Source: P. Christiansen et al., *DeepAnomaly: Combining Background Subtraction and Deep Learning for Detecting Obstacles and Anomalies in an Agricultural Field*, *Sensors* 16(11):1904 (2016), https://doi.org/10.3390/s16111904.

Run `python replot_published_agriculture.py` with Matplotlib installed. Outputs are PDF, PNG, SVG. Numbers are also recorded in `published_measurements.csv`. The book's Figure 5.28 must be captioned **replot of published measurements**, not an independently executed detector benchmark.
