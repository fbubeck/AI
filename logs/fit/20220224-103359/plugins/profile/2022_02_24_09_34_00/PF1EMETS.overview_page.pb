?&$	
??]??NDu?????ʿ?W???!?"?????$	?????@J??????? ?,W?@!-????@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???!?Q???hV?y??A?%?"???YM?Nϻ???rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0J?i?W????mnLO??A??$????Y)???^??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0}iƢ???U???@??A??@????Y????????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?{??m????`⏢???A??qo~???Y?????k??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0p????M????bg
??A????O??Y?s}??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0w?*2: ????T?G???A;?p?GR??Y߇??(_??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ʿ?W???w??o???A???5>???Y͔?????rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?0e? ???l??????AZ?b+hZ??Y6?:???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
??!6X8??mV}??b??A?k?}?
??Y???y??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???J
??E?D??2??A??H.?!??Y?k?) ??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails09Q???????4????A?.l?V^??Y?\??ky??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails062;?????[[%??A?7?Q????Y???!??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?b?????@L<???A?YL??Yz?蹅???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?"???????LLb??A}A	]??Y'JB"m???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0s,?????؛?????A??~???YrP?Lۿ??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ӅX????9]???A</?:??Y?????o??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????w????A9?3Lm???Y??2????rtrain 97*	֣p=
??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?i4???!c?????B@)??F????1??ٜ-5A@:Preprocessing2E
Iterator::Root-|}?K???!?Ks2?@)??}?u???1?ţ?B?/@:Preprocessing2T
Iterator::Root::ParallelMapV2?0}?!8??!???d??.@)?0}?!8??1???d??.@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?jQL???!J??=c3Q@)N??;P??1\????*@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??u?B??!???۽x!@)??u?B??1???۽x!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?[Ɏ???!???M?n,@)_(`;???1^?????@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?&P?"??!4N@?2@)?'*?T??1??:?X?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??m????!?[?Ü@)??m????1?[?Ü@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?x??3@I:??gX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$		7?????K?S7??mV}??b??!?`⏢???	!       "	!       *	!       2$	???????\?$?ˤ????5>???!}A	]??:	!       B	!       J$	???????끴6In?͔?????!'JB"m???R	!       Z$	???????끴6In?͔?????!'JB"m???b	!       JCPU_ONLYY?x??3@b q:??gX@Y      Y@q?f?JR@"?	
both?Your program is POTENTIALLY input-bound because 47.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?73.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 