$	B??_(????_????vOjM??!NbX9???$	???V?@?x?,i@h?
l?!@!}?(5?0.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Q???ޓ??Z???A?A?f????Y~??k	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&NbX9???!?lV}??A<?R?!???Y䃞ͪϥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8EGr???c?ZB>???A?@??ǘ??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??W?2????_vO??A)\???(??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*:??H??L?
F%u??A	?c???YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????ׁ?????????A??????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q?-?????:M??A???S????YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???z6??ݵ?|г??A?rh??|??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?5?;N??9??v????A?{??Pk??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	7?A`?????O??e??AbX9????Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
jM???????????A؁sF????Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??~j?t?????&??A?q?????Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??%䃞???-?????AC??6??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&v????????j+????A??K7?A??Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???_?L?J??AǺ????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????o?? o?ŏ??A2U0*???Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q????|?5^???A?!?uq??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+??n4??@???Au????Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|гY???^K?=???Au?V??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?	??????ׁs??A8gDio???Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM??U0*?д?A?x?&1??Y
ףp=
??*	hffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?n?????!ZdD?q?A@)??	h"l??1?8?W?>@@:Preprocessing2F
Iterator::Model"??u????!????lA@)Zd;?O??1???/?5@:Preprocessing2U
Iterator::Model::ParallelMapV2O@a?ӻ?!'?a??+@)O@a?ӻ?1'?a??+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??Pk?w??!<????IP@)??^??1?YO)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate$(~????!h?϶l+@)lxz?,C??1?Ў??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???V?/??!?V??@)???V?/??1?V??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!1?@?h1@)?<,Ԛ???1?ú[??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*'???????!*???&?@)'???????1*???&?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9;???U@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	x?Wb??????W?1??U0*?д?!!?lV}??	!       "	!       *	!       2$	?/.?'???s)?U????x?&1??!<?R?!???:	!       B	!       J$	?qt?)????a??5i???&S???!~??k	???R	!       Z$	?qt?)????a??5i???&S???!~??k	???JCPU_ONLYY;???U@b 