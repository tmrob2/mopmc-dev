#!/bin/bash

#/build/mopmc --help
./build/mopmc --prism examples/dive_and_rise/dive_and_rise.nm --props examples/dive_and_rise/dive_and_rise_prop_100.props --fn mse --popt away-step

#./build/mopmc examples/multiple_targets/multiple_targets.pm examples/multiple_targets/multiple_targets_21c.props
#./build/mopmc examples/multiple_targets/multiple_targets.pm examples/multiple_targets/multiple_targets_prob_4.props
#/home/guoxin/Downloads/storm/build/bin/storm --prism examples/multiple_targets/multiple_targets.pm --prop examples/multiple_targets/multiple_targets_4.props

#Experiment
#./build/mopmc examples/warehouse_tests/wh-5-2-2.nm examples/warehouse_tests/whouse_tasks.pctl