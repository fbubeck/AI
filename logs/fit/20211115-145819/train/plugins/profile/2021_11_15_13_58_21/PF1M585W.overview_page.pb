?($	@???????z???????v??????!b??4?8@$	*?-])	@???t?@]??\>??!??5q8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$P?s????[ A???AmV}??b??YM?O????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U??v??????A?ׁsF???Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z6?>??!?lV}??Ash??|???Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_??RI??&???A?a??4???Y??	h"l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?MbX9??46<???A?Q???Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????,Ԛ????A??ͪ????Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?
F%u??m???????A?3??7??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=,Ԛ?????v????A8gDio???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&n??@??V?/???AM??St$??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	b??4?8@??ܵ??A?d?`TR??Yŏ1w-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ꕲq?????镲??Ad?]K???Y??ܵ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e?c]??%??C???A?J?4??Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ׁsF?????????A??C?l??Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?
F%u??KY?8????A????Y?5?;Nѡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??a??4??????(??AV-???Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l????q????A?J?4??Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#J{?/L????:M???A?Q?|??YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5????7??d??A$(~??k??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5??X9??v???A?46<??Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5????/?'??A[B>?٬??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v???????c?ZB??A?rh??|??Y??~j?t??*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?[ A???!?M????@)??7??d??1?=%??<=@:Preprocessing2F
Iterator::Model ?~?:p??!?kU E@)??x?&1??1?w?Lϼ9@:Preprocessing2U
Iterator::Model::ParallelMapV2x$(~??!?_??pI0@)x$(~??1?_??pI0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[????<??!r?????L@)?B?i?q??1??#am%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??H.?!??!4?S'@)??H.?!??14?S'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateB>?٬???!? 2?'@)ޓ??ZӬ?1???l??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapףp=
???!?R?>?/@)??A?f??1?@X@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??@??ǘ?!?/߱@)??@??ǘ?1?/߱@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9π+?*?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	????p???xQ???c?ZB??!??V?/???	!       "	!       *	!       2$	??K^?????WpJ???rh??|??!?d?`TR??:	!       B	!       J$	"?D@-0??]?ȶ?֐???~j?t??!M?O????R	!       Z$	"?D@-0??]?ȶ?֐???~j?t??!M?O????JCPU_ONLYYπ+?*?@b Y      Y@q#,"R?D@"?
both?Your program is POTENTIALLY input-bound because 61.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?41.2743% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 