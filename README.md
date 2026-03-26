# Whisker array sensing simulator

## Headless mode

Headless mode is enabled with the `--headless` option. It is meant for agent play and file rendering. An animation can be optionally saved with `--save-file <name.mp4>`.

## Replay mode

The replay mode is enabled with the `--replay <replay_data_file>.csv` option. 
The simulator recreates the movement of the whisker array from a replay file, which should be a csv file with 5 columns: time, x, y, x_vel, y_vel.

This mode can be used to create animation of saved agent runs. The following is an example command to create animation from a replay file.

```shell
python whisker_array_simulator.py --data-path <flow-data-path> --dynamic-flow --flow-dt 0.04 --flow-time-delay 3  --flow-decay 10 --replay <replay_data_file>.csv --headless --save-file <replay_file_name.mp4|gif> --save-dpi 150
```

The `--dynamic-flow` option enables dynamic flow using all flow data frames in the `<flow-data-path>`. The `--flow-decay` is used when the run time is beyond what's available in the flow data. The flow velocity amplitude is set to decay with the set half-life (10s in the above example).

Note that to recreate the replay faithfully, be sure to set the `--flow-time-delay` to the same as the original run.

