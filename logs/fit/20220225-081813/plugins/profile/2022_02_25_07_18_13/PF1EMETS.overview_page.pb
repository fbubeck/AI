?&$	???˺???T&=9o???O??'?9??!R?8?ߡ??$	?(Y9?%	@??6R?&???Ev@!??d?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??	h"l???2??Y??AR?????Yy?ѩ+??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??x?????-"??`??A????a???Y????x̐?rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0l???l??k??qQ-??A??\7????Y28J^?c??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0\u?)I??KZ?????A???Y.??Y9?)9'???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G?Z?Q???.:Yj????A??&????Y|??????rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?q?@H??????iO???A&U?M?M??Y??G????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??l??3???g@???An??????Y?????x??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	R?8?ߡ??8?ܘ????AC9ѮB??YI??? ??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
R???<??K???>??A???????Y??oa?x??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ȗP?a???t??????Ae??????YN???
a??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?oa?x????D.8????A^??-???YD?o֐?rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???LL????qQ-"??A?OVW??Y?K?b??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???o?????9????A?6?????Y`X?|[???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0(??Z&C?????????A?E?>???Yկt><K??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0O??'?9??^??????Ag?????Y?8??m4??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??I????>?#d ??A[z4Փ??Y'??d?V??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?s???????VC???A??s?????Y???.5B??rtrain 97*	???(\_@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeate???}??!?g<?C@)??[v???1?????A@:Preprocessing2T
Iterator::Root::ParallelMapV2??^???!??b?B0@)??^???1??b?B0@:Preprocessing2E
Iterator::Root$???E???!W??4@@)?\p???1?qIg%0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip'???????!T???P@),-#??ʩ?19?y?'$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice>?h??!?#??#@)>?h??1?#??#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?=&R?ͳ?!??3S?.@)??x?@e??1"?c??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap<??J"???!??*??p3@)??????1E?AF@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*P?c*???!?ĩ???@)P?c*???1?ĩ???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 42.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9KVC?t	@IN??X?7X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	m&?c???xY?X???^??????!8?ܘ????	!       "	!       *	!       2$	(GX??V???6j????e??????![z4Փ??:	!       B	!       J$	??U?K?????
?!0c?y?ѩ+??!??oa?x??R	!       Z$	??U?K?????
?!0c?y?ѩ+??!??oa?x??b	!       JCPU_ONLYYKVC?t	@b qN??X?7X@Y      Y@q?l???R@"?	
both?Your program is POTENTIALLY input-bound because 42.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?74.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 