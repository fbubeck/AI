?)$	?z6?>??6'aw:??????JY???!a2U0*???$	N?ߧm?@?M?31@??/?_?@!?????9@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?8??m4???c?ZB??A4??7????Y?^)????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}???? ?	??A}??b???Ye?`TR'??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d?]K?????	h"l??A??z6???Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI???Q?|a2??A"??u????Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c???ڊ?e???Ae?X???Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio???U???N@??AF%u???YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z6?>??%??C???A???????YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ???????:M???An????Y??(????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????j?t???A??h o???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	Nё\?C??????߾??Ao?ŏ1??Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
<Nё\?????????Ad?]K???Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??@?????z6?>??A6<?R?!??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'?????K7???A??Q????Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7?????JY?8??A????????Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??	h???HP???AC??6??Y????镢?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*???	?^)???A???T????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^??Ϊ??V???A?? ?rh??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?I+????2ı.n??A?JY?8???YW[??재?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx?????(???A6?;Nё??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Pk?w??Nё\?C??A?lV}???Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY???Ș?????Aꕲq???Y??ݓ????*	???????@2F
Iterator::Model?	?c??!? az?8I@)?:M???1?YJ??6C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??(???!.ܜ??8@)??Pk?w??1Wu?7@:Preprocessing2U
Iterator::Model::ParallelMapV2-!?lV??!_[?

(@)-!?lV??1_[?

(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<Nё\???!*ߞ?=?H@)??H.???1f?ނF$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?H?}8??!??-$?(@)?H?}8??1??-$?(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?0?*???!s??\;0'@)?s????1?&T??7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??B?i???!?5???$-@)0L?
F%??1?!݄??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*a??+e??!&!lVN???)a??+e??1&!lVN???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t49.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?`???
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?o S????:?)???Ș?????!	?^)???	!       "	!       *	!       2$	Y<?}??IQ?????ꕲq???!?? ?rh??:	!       B	!       J$	?R?	ɩ???]^u5??L7?A`???!?^)????R	!       Z$	?R?	ɩ???]^u5??L7?A`???!?^)????JCPU_ONLYY?`???
@b Y      Y@q?<۠?G:@"?	
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t49.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?26.2794% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 