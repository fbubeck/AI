?)$	??E??=???ka??d??Ș?????!?W?2?1@$	TF?E_@,ڤ{P@l????@!n ????<@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?W?2?1@????S??A+?????Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4???? ?rh??A??S㥛??Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l???:??H???A??+e???YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?rh??=?U?????A2w-!???Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(?????????B??A?s?????Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ??7?[ A??A???ZӼ??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??v??????QI????A???x?&??YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????h"lxz???A?]K?=??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a2??j?q?????AB>?٬???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??H.?!??_)?Ǻ??A"?uq??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??6???Ǻ????A?-???1??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??u????c?ZB>???A??QI????Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z?):???????S????A\ A?c???Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& c?ZB>??]m???{??A?*??	??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?a??4???,Ԛ????Aŏ1w-!??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????lV}???A?	???Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%u???&S??:??A?٬?\m??Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?G?z???s?????A      ??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f???.?!??u??A+??ݓ???Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\??S??:??A/?$???Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș???????q????Ar?鷯??Y??_?L??*	???????@2F
Iterator::Model???N@??!????9?K@)S?!?uq??1?ؼ?P:H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat<Nё\???!?>N???8@)?ܵ?|???1?Sr`@z4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX?5?;N??!a?F@)m????ҽ?1?Z?x?@:Preprocessing2U
Iterator::Model::ParallelMapV2u?V??!??`?G@)u?V??1??`?G@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?[ A???!w豻?@)?[ A???1w豻?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??m4????!4?o??F@)??m4????14?o??F@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?o_???!Zi.R?!@)??ܵ???1}?U?y@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap46<?R??!e?P??'@)??ZӼ???1+h????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t47.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???B+@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	+?T"???????J????q????!????S??	!       "	!       *	!       2$	???)?????w_?:???B>?٬???!+?????:	!       B	!       J$	?0gS??i'[?g??L7?A`???!??????R	!       Z$	?0gS??i'[?g??L7?A`???!??????JCPU_ONLYY???B+@b Y      Y@q,ݮ\?FA@"?	
both?Your program is MODERATELY input-bound because 7.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t47.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?34.5528% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 