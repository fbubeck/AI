?&$	?,?????i????????71$'???!_
?]???$	p???߽@?ʵI?w????qZ(@!???[1@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Cl0???F??1???A?zj????Y"lxz?,??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?7k???????0Xr??A?(ϼv??Y ???Qc??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0q㊋????y????A??L????Y'??2???rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0_
?]???`?_????Ah>?n???Y??o??R??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?$???}?????|y??A?/??L???Y5??a0??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails05ӽN?????Y/?r???A??????Y??*ø??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?71$'???O;?5Y???A???{??Yd:tzލ??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?ơ~??????K???Ab֋??h??Y?f?\S ??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
.9?????jkD0.??Ag{?????Y~R??????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??(?????[?tY??A?#??:??Y?*?MF???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???u???????????AFCƣT???Y???i????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???C???????Z???Aq8??9@??Y?V?Sb??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??????u/3l???A???????Y:?w????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0Ĳ?C???o?$?j??A????@???Y??U]??rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ke?/?s??Ll>???A?9????YT??Yh???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???R???R)v4???Aof??????Y?W??V???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0|?_??????|A??A??????Y??7????rtrain 97*	33333{?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?Hg`?e??!?jW%??B@)?E??????1T?jC%/A@:Preprocessing2E
Iterator::Root̲'??9??!l*e???@@)&䃞ͪ??1?????2@:Preprocessing2T
Iterator::Root::ParallelMapV2s??c?ȸ?!?x?!?-@)s??c?ȸ?1?x?!?-@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???,???!?jM???P@)?? ?S???1??͉??#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???sӮ?!)?u?b_"@)???sӮ?1)?u?b_"@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?խ??޿?!?????2@)v?1<???1??р?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???u???!?vA?o*@)-`?????1G.4?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?e3???!???Nx@)?e3???1???Nx@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9W?+<??@IE?VJ#X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??׽l??????x???Ll>???!`?_????	!       "	!       *	!       2$	d??C??Է?u?'?????{??!h>?n???:	!       B	!       J$	??????????U?v??*?MF???!??o??R??R	!       Z$	??????????U?v??*?MF???!??o??R??b	!       JCPU_ONLYYW?+<??@b qE?VJ#X@Y      Y@qZ
EL@"?	
both?Your program is POTENTIALLY input-bound because 43.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?56.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 