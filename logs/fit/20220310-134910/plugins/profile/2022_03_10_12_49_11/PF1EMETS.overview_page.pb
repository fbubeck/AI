?&$	F?գf??t?????`s?	M??!G??$	e????@?;A?1????_??@!@Ĉ?y?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0%;6??????? ?X??A5c?tv2??Y0?[w?T??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0/M??????̰Q?o??A??"???Y??x"????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??^b,????a?Q+L??A?<HO?C??Yi??ᴐ?rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0#???????
???I'??A????????Y??J?????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??/ע??G?,????AJ??%?L??YM0?k????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ZF?=??????	/????Aa?N"¿??Y |(??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G9?M?a??}гY????AF??}???Y?C4??ؙ?rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	G??` ?C????A???,?s??Yv4?????rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
??0E?4??I/j?? ??A???	???YN+?@.q??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????':???????Aa3?ٲ??Y?g????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0'i?????????Ax?ܙ	???Yҩ+??y??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????ƃ-v???A?*q???Y?6?ُ??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0`s?	M????g?RD??AC9ѮB??Yhv?[????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??v????p?n?????A?衶??Yx??Dg???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$`tys8??*q㊋??A???7/N??Y5~??$ϕ?rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C??g????	?c??A??lu9%??Yђ?????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C8???(?XQ?i??A)u?8F???Y?4F??j??rtrain 97*	??Mb[?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??@J????!????A@)?A	3m???1?.3?W?@:Preprocessing2E
Iterator::Root??V??,??!c?jR"?B@)??Cl?p??1]~Z[ɻ3@:Preprocessing2T
Iterator::Root::ParallelMapV2??i? ???!i?zI{?1@)??i? ???1i?zI{?1@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipԚ?????!?T???5O@)?Xİè?1???ޚ{"@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice
???%???!a	z?!@)
???%???1a	z?!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??R??!:1 X??-@)+n?b~n??1?O*? ?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapj???늹?!O?֓L3@)=?K?e???1???=g@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?Bus????!0??%).@)?Bus????10??%).@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??w.?@InB?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??m??????k;^????	?c??!
???I'??	!       "	!       *	!       2$	?S7?un???^<M;??C9ѮB??!???,?s??:	!       B	!       J$	???#?U?? 4?	?l?ҩ+??y??!?g????R	!       Z$	???#?U?? 4?	?l?ҩ+??y??!?g????b	!       JCPU_ONLYY??w.?@b qnB?X@Y      Y@q??a?U@"?	
both?Your program is POTENTIALLY input-bound because 45.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?87.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 