$	?H#?D???A??????k+??ݓ??!??d?`T??$	5?g
$?@??Y???@???1?H@!???6?4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??a??4??؁sF????AX?5?;N??YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????.???1???AA??ǘ???Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F????a??4??AK?46??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S???-?????AOjM???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ???? ?	???A c?ZB>??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ????????<,??AF??_???Y?D???J??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?j+???????{??P??A?.n????Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?|?? A?c?]??A46<?R??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)??Y?? ???AA?c?]K??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	ꕲq???7?A`????A??ͪ????Y?A`??"??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?l??????9??v????AR'??????Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$???~????? ?rh??AS??:??Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM???6?[ ??A4??@????Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	?????e??a??AǺ?????Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"??_?L?J??AW[??????Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%??	?^)???A???1????Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e????QI??&??A?D?????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??d?`T???A`??"??A??????Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??ݓ????|гY???Aףp=
???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5?????{??P??A?Y??ڊ??YI.?!????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ???3??7???A?`TR'???Yvq?-??*	33333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeataTR'????!޶FrpB@)Tt$?????1G?x??@@:Preprocessing2F
Iterator::Model??e??a??!g?o:BC@)&S????1Xd ?7@:Preprocessing2U
Iterator::Model::ParallelMapV2??x?&1??!?p???*-@)??x?&1??1?p???*-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?^)???!??3?ŽN@)??????1؞M?="@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?W[?????!C?vu?@)?W[?????1C?vu?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate	?c???!???S?A(@)Ǻ?????1?)1Ѩ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*? ?	???!br?h@)? ?	???1br?h@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9EGr???!Lg?K?.@)??Ɯ?1Z?ڡ?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??4o?@@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	\??0p???;=??l???3??7???!?A`??"??	!       "	!       *	!       2$	???#??????̴Z???X?5?;N??!?D?????:	!       B	!       J$	OCA?x???5?n?;???&S???!U???N@??R	!       Z$	OCA?x???5?n?;???&S???!U???N@??JCPU_ONLYY??4o?@@b 