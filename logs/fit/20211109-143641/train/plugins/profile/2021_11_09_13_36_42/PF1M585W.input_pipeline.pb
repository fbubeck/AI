$	?~+?????G(u?7????.n????!}?5^?I??$	0?@?????@ ??????!;?;?+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?z?G?????|?5^??AQ?|a??Y??@??Ǩ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h???C?l????Au?V??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C??@?߾???A??????Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?/L?
??t??????A??0?*??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??0?*???镲q??A???&??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I??5^?I??A?i?q????Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&      ??Dio?????Av??????YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"lxz?,??????Q??A??????Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?'???:M???Aa??+e??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	????o??46<?R??AC?i?q???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
5?8EGr????ǘ????A#??~j???YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????!??u???A????x???YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??镲??X9??v??A?8EGr???Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&䃞ͪ????ͪ??V??A<?R?!???Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&r?鷯????4?8E??A0?'???Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C???? ???AEGr????Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[??6?;Nё??A??D????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș????????H??A???Q???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c??c?=yX??A~8gDi??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(???EGr????A7?A`????Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n????]?C?????A??ܵ?|??Y??ܵ?|??*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!?/??v'@@)?\?C????1u{$V??=@:Preprocessing2F
Iterator::Model??_?L??!!?W$?NB@)??1??%??1 C.owR5@:Preprocessing2U
Iterator::Model::ParallelMapV2鷯???!?B???.@)鷯???1?B???.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?s????!???#?O@)?=yX???1+BR???&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?:pΈҮ?!+?jnXO @)?:pΈҮ?1+?jnXO @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?ŏ1w??!?d3//@)?Q???1~??E?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?:pΈ??!??^??3@)???B?i??1ؠ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Dio??ɔ?!F$}??@)Dio??ɔ?1F$}??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???d@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??M#???b??P??????|?5^??!5^?I??	!       "	!       *	!       2$	?*ՃE???W5???????ܵ?|??!?8EGr???:	!       B	!       J$	5?A?}????????w??????Mb??!??@??Ǩ?R	!       Z$	5?A?}????????w??????Mb??!??@??Ǩ?JCPU_ONLYY???d@b 