?&$	?%{7????vj?????` ??c??!)$??;???$	?Q6??@?9?????\?w???@!?e\6&@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?.?ꏰ??I?2????A????th??Y???`U??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0v?Kp???1%??e??AQ?O?I???Y??[;Q??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0j???Z?????(@??A?1<??X??Y??Ȯ????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0g??Ry??ж?u????A?u??S??Y?B??f??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$
-??1????ǘ????A]?&????Y????Dh??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails02*A*???9???`??AT7???Y?;???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0+????h?o}Xo??A???????Y?! _B??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	` ??c???u??O??Aa?$????Yr5?+-#??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
$?6?D????QH2?w??A\????o??Y??+????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0F%u?????)????A?k&?ls??Y?X?_"ޚ?rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0)$??;????OU?????AYQ?i>??Y??f?v???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0L?{)<????4LkS??A?ܘ?????Yz?rK???rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Y?+?????	??ϛ??A???DR??Y??f?R@??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0`?? @????*????A??\5???Y"ĕ?wF??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0t?f??(??J?y???AB????W??Y
?????rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??!?Q????*?WY???A?n?|?b??Y??SW>??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????Z?{c??AU?Y??Y?_cD???rtrain 97*	ףp=??@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???u?|??!1?? ?C@)JӠh???1?WDMM?A@:Preprocessing2E
Iterator::Rootp]1#?=??!*? ???;@)??
??X??1?2?v?8-@:Preprocessing2T
Iterator::Root::ParallelMapV2??W??"??!??}9?t*@)??W??"??1??}9?t*@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??\??$??!?i?54@)?a??c??1 ???<?%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip(???I???!u???L
R@)?t?? ??1????&{#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????˽?!??/?ޥ"@)????˽?1??/?ޥ"@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?Ӹ7?a??!$??B8@)???1???1m??6@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?ȑ??ț?!>*?7}c@)?ȑ??ț?1>*?7}c@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??@I???~??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	To?b(??oY?A!????u??O??!?OU?????	!       "	!       *	!       2$	bBk?Y?????Ӌ???a?$????!YQ?i>??:	!       B	!       J$	??ݠ????k?"????;???!??f?v???R	!       Z$	??ݠ????k?"????;???!??f?v???b	!       JCPU_ONLYY??@b q???~??W@Y      Y@qԖЬ?H@"?	
both?Your program is POTENTIALLY input-bound because 43.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?49.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 