$	gfffff??x
V+`????????!?q??????$	Kg]??@̨???
@*yJ'7@!
?"(??,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$$???~???4??@????AS?!?uq??Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K???NbX9???AˡE?????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~??k	????G?z???A鷯???Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?V?????H??A??	h"??Y?A`??"??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ???io???T??A?鷯??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~??8??d?`??A???????Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?%䃞?????6???A??h o???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q??????I.?!????Aݵ?|г??Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??#?????䃞ͪ???A?w??#???Y?MbX9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?ZӼ???????K7??A?鷯??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??ZӼ???D????9??A?HP???YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?B?i?q??? ?	???AT㥛? ??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??k	????F%u???A?c?ZB??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?k	??g??l	??g???AE???JY??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[????<??*:??H??A??ڊ?e??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=?U?????4??7????A=
ףp=??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????B????K7?A`??A???S????Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l??}?5^?I??A؁sF????Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"lxz?,??J{?/L???AU0*????YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Fx$?????1????Ao???T???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????ׁsF??A+?پ?Y?]K?=??*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??H?}??!f???8*A@)??St$???1@??J ?@:Preprocessing2F
Iterator::Model???T????!????3Y@@)???ZӼ??1s???^3@:Preprocessing2U
Iterator::Model::ParallelMapV2??^)??!7??U??*@)??^)??17??U??*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateH?z?G??!???)?0@)?0?*??1EYhKo?$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0L?
F%??!<??f?P@)333333??1???"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice	?c???!?=W?m@)	?c???1?=W?m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap      ??!?ͮd?7@)?z?G???1???`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?A`??"??!dv?;?
@)?A`??"??1dv?;?
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9^??^??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?"N?????9??ab????ׁsF??!I.?!????	!       "	!       *	!       2$	?(?[??????????+?پ?!?w??#???:	!       B	!       J$	b?I&??????Ɏ??????H?}??!p_?Q??R	!       Z$	b?I&??????Ɏ??????H?}??!p_?Q??JCPU_ONLYY^??^??@b 