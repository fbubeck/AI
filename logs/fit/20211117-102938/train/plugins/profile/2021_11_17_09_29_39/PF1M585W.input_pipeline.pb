$	?>J"????+k??????9#J{??!???Mb??$	p?Jף?@2???	@?^?s?f@!X?s1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$D?l?????-??臨?A?H?}8??Y o?ŏ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????9#??ŏ1w-??A\???(\??Y?lV}???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????M????<,Ԛ??Az6?>W[??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M??? o?ŏ??A?]K?=??Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q?-???J+???A?-?????YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>??????K7???A??????Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J+???z6?>W??A?:pΈ???YbX9?Ȧ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb??X9??v??A?=yX???Y"?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?????1?*????A?;Nё\??Y??|?5^??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??	h"????ݓ????A]?C?????Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
P??n???:??H???Aj?t???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z?):?????;Nё\??A"?uq??YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?HP??????????A)?Ǻ???YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????B?i???A???镲??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?q?????_)?Ǻ??A???&??Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]??X?5?;N??A??4?8E??Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O????4?8EG??A????Mb??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb??h"lxz???A?q?????Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????a2U0*???A?O??n??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ??[????<??A??D????YQ?|a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{????Pk?w??A?ܵ?|???Ya2U0*???*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??q????!br??A@)?ZӼ???1쌽C?2@@:Preprocessing2F
Iterator::ModelC??6??!?,?>-&B@)??W?2???1????6@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1???!????i?*@)y?&1???1????i?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???T????!?K???O@)?	h"lx??1 ?|n?'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??\m????!??.T??@)??\m????1??.T??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??y?):??!k4???'@)?R?!?u??1?m?T?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???N@??!Ko5c?10@)1?Zd??1UT?㹰@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ʡE???!n?H??u	@)??ʡE???1n?H??u	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9HY?ִ$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	:??ɭ????"? ??-??臨?!:??H???	!       "	!       *	!       2$	??ϒE]??*b^E????ܵ?|???!?=yX???:	!       B	!       J$	????b???v?Q??B>?٬???! o?ŏ??R	!       Z$	????b???v?Q??B>?٬???! o?ŏ??JCPU_ONLYYHY?ִ$@b 