$	e}?????Y4؄ B???٬?\m??!???T????$	????u@t???Po@~??7 @!-??#EC,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Q???????b?=??A?/?$??Y??@??Ǩ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????"??~j??A?_?L??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??v?????ׁsF???A=?U????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gx$(???鷯??A?ǘ?????Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B?f??j?????h o??A?-?????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;??o?ŏ1??A?"??~j??Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾???]?Fx??Aݵ?|г??YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx??A??ǘ???A?9#J{???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?u???????z6???A?Fx$??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	L?
F%u????j+????A??Q???Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?????????s????A?'????Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[Ӽ?????0?*??A??S㥛??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??u?????lV}????ADio?????Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&KY?8?????h o???A??#?????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|?5^??????&S??A?j+?????YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I????????A?h o???Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?{??Pk???z?G???An????Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????m???????A??@?????YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ??7?[ A??A?t?V??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ??_)?Ǻ??A??镲??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?٬?\m??P??n???A?z6?>??Y???<,Ԛ?*	fffff6?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	?c?Z??!????A@)w??/???1?U??i?@:Preprocessing2F
Iterator::Model??ʡE???!??95\B@)m???{???1?pu?N<7@:Preprocessing2U
Iterator::Model::ParallelMapV22w-!???!?????*@)2w-!???1?????*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?HP???!-????O@)k+??ݓ??1?XJϾ?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?G?z??!?=??&,@)"?uq??1??9]?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???߾??!??4!?!@)???߾??1??4!?!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS?!?uq??!æ???2@)"??u????1????g@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*$????ۗ?!??8??8@)$????ۗ?1??8??8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	 ?P?ֲ??\?k%???P??n???!m???????	!       "	!       *	!       2$	???S????p??????z6?>??!?j+?????:	!       B	!       J$	??uS5*?????ӟ??q??????!??@??Ǩ?R	!       Z$	??uS5*?????ӟ??q??????!??@??Ǩ?JCPU_ONLYY??????@b 