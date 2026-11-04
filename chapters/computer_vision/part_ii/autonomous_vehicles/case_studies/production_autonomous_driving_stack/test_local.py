"""Fixture-only trace-statistic checks; not production evidence."""
import csv,tempfile
from pathlib import Path
from evaluate_pipeline_trace import FIELDS,load_trace,summarize
def main():
    with tempfile.TemporaryDirectory() as d:
        file=Path(d)/'fixture.csv'
        with file.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(FIELDS)
            for j in range(30):w.writerow([j,j*.1,20,15,10,5,3,6,1,0])
        result=summarize(load_trace(file),100)
        assert result['frames']==30
        assert result['latency_p95_ms']==45.
        assert result['deadline_miss_fraction']==0.
        assert result['emergency_brake_command_fraction']==0.
    print('PASS: original trace schema and latency quantile algorithms')
if __name__=='__main__':main()
