?($	q?\P???????k???u?V??!?q??????$	??9?}@(D$?k?@[???????!??ǨT,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?T???N????HP??A?y?):???Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?#????????q????AU???N@??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????G?z???A%u???Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?a??4?????4?8E??A?:M???Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????H??0?'???A?Pk?w???YM?J???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&R???Q???H.?!???Aı.n???Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;?O??n???O??n??AEGr????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46???
F%u??A?X????YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`???O@a????A??ܵ??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???QI????K7?A`??A?W?2ı??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
5?8EGr??$(~??k??A??????Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???x?&????1??%??A?q?????YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I??Ǻ????A/?$????Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5???c?]K???A?k	??g??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;M?O??!?rh????A??e?c]??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??St$????H.?!???An4??@???YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?+e?X??I.?!????A0?'???Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.???u?????A???&S??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ϊ??V???鷯???ApΈ?????Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q???????R?!?u??A?4?8EG??YH?z?G??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?V????a??4??A?W?2ı??Y2??%䃞?*	gffffF?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??%䃞??!9??^_@@))\???(??17`???>@:Preprocessing2F
Iterator::Modell	??g???!&B?U`A@)j?t???1My
??t4@:Preprocessing2U
Iterator::Model::ParallelMapV2ڬ?\mž?!?	?V??,@)ڬ?\mž?1?	?V??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<?R?!???!m?^!?OP@)h??|?5??1?^??,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceY?? ޲?!7`????!@)Y?? ޲?17`????!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????!͕[??,@)??	h"l??1-kXS?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMappΈ?????!?A??rx2@)z6?>W[??1w?!?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?+e?X??!ԁqj??@)?+e?X??1ԁqj??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?<??XU@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	8???@???8Q?h?????HP??!鷯???	!       "	!       *	!       2$	??KF?2??K?v???W?2ı??!?4?8EG??:	!       B	!       J$	??E<????T?K???W[?????!H?z?G??R	!       Z$	??E<????T?K???W[?????!H?z?G??JCPU_ONLYY?<??XU@b Y      Y@q????_@@"?
both?Your program is POTENTIALLY input-bound because 51.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.7468% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 