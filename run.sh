#!/bin/bash

# Storm Test
#./build/mopmc examples/multiobj_scheduler05.nm examples/multiobj_scheduler05.pctl
#./build/mopmc examples/multiobj_consensus2_3_2.nm examples/multiobj_consensus2_3_2.pctl
#./build/mopmc examples/multiobj_scheduler_x.nm examples/multiobj_scheduler05.pctl
#./build/mopmc examples/multiobj_scheduler05.nm examples/multiobj_scheduler_max.pctl

./build/mopmc examples/multiple_targets/multiple_targets.pm examples/multiple_targets/multiple_targets_4.props
#/home/guoxin/Downloads/storm/build/bin/storm --prism examples/multiple_targets/multiple_targets.pm --prop examples/multiple_targets/multiple_targets_4.props


#Experiment
#./build/mopmc examples/warehouse_tests/wh-5-2-2.nm examples/warehouse_tests/whouse_tasks.pctl