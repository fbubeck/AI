?&$	??&@x???CgA??W?'???!m?????$	??p?@? )?7????:b??@!?A???@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????????"??^??A?~T???Y???e??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????1????$W@??A??p???Yl?˸???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?3??E????%Z?x??A* ??q??Y?o|??%??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0]m??????6>???4??Ay\T??b??Y? 4J????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G?J??q????0{?v??A?.?????Y?SW>????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0!??F???????l???A?^??x???Y?	Q???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0W?'????}V?)??A??[?O??Y(???%V??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	yY|??????.??A^??a?Q??YD1y?|??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
Q???`E????hW!???A?????`??Y????N??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??E?nt??'?y?3??Ad??u??Y
???????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0x(
??<??ٕ??zO??A||Bv????Y????+??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0pΈ??????%9`W???A??????YYni5$???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?4?8EG??
?2?&??A?#??S ??YH?Sȕz??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0m?????Ý#????A???r???Y/???uR??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?;? ?????@???A??`U????Y????4???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0d????t??ڌ?U???A?UfJ?o??Y?$xC??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???n,(??eT?? ??A<hv?[???Y*?~?????rtrain 97*	??S㥞?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat ?&?f??!"??Æ?D@)3?뤾,??1?l?ԝC@:Preprocessing2E
Iterator::Root?\5???!?dn$`?@)?4*p???1N#)|?S1@:Preprocessing2T
Iterator::Root::ParallelMapV20Ie?9??!??v??,@)0Ie?9??1??v??,@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip'?W???!??f??'Q@)?F??ұ?1V_ ??<#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????????!?T???"@)????????1?T???"@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?>?6???!V-ܗk?+@)ܻ}????1???(;@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???"M???!?c 1@)?\???1?:? j?
@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?,{؜??!??Q!+@)?,{؜??1??Q!+@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?E?Fo@Iѵȅ?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?`??
??\?
)J????}V?)??!?"??^??	!       "	!       *	!       2$	?0)?G???w9?Ӧ????[?O??!???r???:	!       B	!       J$	??8L?m??*????p??	Q???!/???uR??R	!       Z$	??8L?m??*????p??	Q???!/???uR??b	!       JCPU_ONLYY?E?Fo@b qѵȅ?X@Y      Y@q?M?N
xQ@"?	
both?Your program is POTENTIALLY input-bound because 46.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?69.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 