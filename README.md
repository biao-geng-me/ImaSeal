# Whisker array sensing simulator

## Headless mode

Headless mode is enabled with the `--headless` option. It is meant for agent play and file rendering. An animation can be optionally saved with `--save-file <name.mp4>`.

## Replay mode

The replay mode is enabled with the `--replay <replay_data_file>.csv` option. 
The simulator recreates the movement of the whisker array from a replay file, which should be a csv file with 5 columns: time, x, y, x_vel, y_vel.
In headless replay mode, if `--save-file` is not provided, output is saved next to the replay file using the same base name with `.mp4` extension by default; if mp4 encoding is unavailable, it falls back to `.gif`.

This mode can be used to create animation of saved agent runs. The following is an example command to create animation from a replay file.

```shell
python whisker_array_simulator.py --data-path <flow-data-path> --dynamic-flow --flow-dt 0.04 --flow-time-delay 3  --flow-decay 10 --replay <replay_data_file>.csv --headless --save-file <replay_file_name.mp4|gif> --save-dpi 150
```

Replay CSV units are configurable and are converted to simulator SI units internally.
Defaults preserve current behavior:

- `--replay-time-unit s`
- `--replay-pos-unit m`
- `--replay-vel-unit m/s`

Supported values:

- replay time unit: `s`, `ms`
- replay position unit: `m`, `mm`
- replay velocity unit: `m/s`, `mm/ms`, `mm/s`

Example for files where `t` is in milliseconds and `x,y` are in millimeters:

```shell
python whisker_array_simulator.py --data-path <flow-data-path> --dynamic-flow --flow-dt 0.04 --flow-time-delay 3 --flow-decay 10 --replay <replay_data_file>.csv --replay-time-unit ms --replay-pos-unit mm --replay-vel-unit mm/ms --headless --save-file <replay_file_name.mp4|gif> --save-dpi 150
```

The `--dynamic-flow` option enables dynamic flow using all flow data frames in the `<flow-data-path>`. The `--flow-decay` is used when the run time is beyond what's available in the flow data. The flow velocity amplitude is set to decay with the set half-life (10s in the above example).

Note that to recreate the replay faithfully, be sure to set the `--flow-time-delay` to the same as the original run.

