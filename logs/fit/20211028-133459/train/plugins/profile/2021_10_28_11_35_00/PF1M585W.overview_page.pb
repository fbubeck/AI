?($	5trl???&?"?????+????!?ׁsF???$	j?!2>@r???y@???+٫@!{??WΠ/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$h"lxz???C??6??Aa??+e??Y???H.??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ׁsF???r?鷯??ANё\?C??Y?'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??????ʡE????A?`TR'???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY????????9#??Aa??+e??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[????<??}гY????Aףp=
???Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ????H?}??A?1w-!??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*:??H??_?Q???A??|?5^??Y?????K??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&.?!??u??O??e???A??????Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$????ףp=
???A?J?4??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	????9#???????Ao???T???Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?%䃞???d;?O????A???1????Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U??ףp=
???AZd;?O??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D?l?????]?Fx??A?JY?8???YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M??OjM???A_?Q???Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u??????(???A??B?i???YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~??k??h??s???A?c?ZB??Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s???????1????A??HP??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?=yX?????????A?5?;N???Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio??ё\?C???A/?$???Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'1?Z???i?q????As??A???Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??????ǘ????A??d?`T??Y??JY?8??*	?????$?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat&S??:??!??IB^jB@)+?????1c??_?.A@:Preprocessing2F
Iterator::ModelM?O????!?%?Ή?B@)?j+?????1|???:,8@:Preprocessing2U
Iterator::Model::ParallelMapV2d;?O????!z>?n??*@)d;?O????1z>?n??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??o_??!"?u1vGO@)??? ?r??1>???D5%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceB>?٬???!?Wz(?#@)B>?٬???1?Wz(?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?'????!Xͼx@&@)46<?R??1?B??g]@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???h o??!?`???.@);?O??n??1nM?Ԋ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*A??ǘ???!H8.%^?@)A??ǘ???1H8.%^?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9W????T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?C	aݢ??7???z??C??6??!?ʡE????	!       "	!       *	!       2$	?W2o????????B????d?`T??!Nё\?C??:	!       B	!       J$	CҔ????T?S??8????~j?t??!?'????R	!       Z$	CҔ????T?S??8????~j?t??!?'????JCPU_ONLYYW????T@b Y      Y@q?q??g?B@"?
both?Your program is POTENTIALLY input-bound because 47.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.5422% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 