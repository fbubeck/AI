$	?x??????O?z2?"??vOjM??!???????$	Du?}??@?~?%@}?q[?@!A????H1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$jM??S??u?V??Ae?X???Y?X?Ѱ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????S???X????A?\m?????Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&S????	?c?Z??A??MbX??YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?#?????????N@??A??g??s??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZB>????>?٬?\??A?I+???YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^?I??"lxz?,??AI??&??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& ?o_???io???T??AS??:??Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S?????^??A???N@??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&.?!??u???B?i?q??A?!??u???Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?.n????? ?rh???A??m4????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?!??u???????K7??A?:M???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&TR'??????߾?3??ApΈ?????YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~??I??&??AM?J???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&333333??Dio?????A.?!??u??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ʡE?????-????AI.?!????YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF??8??d?`??AC?i?q???Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8??d?`??RI??&???A`vOj??Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????!?rh????A,e?X??YEGr????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&I.?!?????z6?>??AHP?s???Y?MbX9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~??k	???j?q?????AX?5?;N??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM??8gDio??A? ?	???Y??ݓ????*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? ?	???!?ŃX?[B@)?? ?	??1??H???@@:Preprocessing2F
Iterator::Model?I+???!<	<?hA@)??<,Ԛ??1S+=5@:Preprocessing2U
Iterator::Model::ParallelMapV2鷯???!Y?ο?'+@)鷯???1Y?ο?'+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??MbX??!?a???KP@)46<?R??1s?sqa?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice????߮?!y?N @)????߮?1y?N @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?G?z??!???C,@)???QI??1?rk?;?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?A`??"??!?,??1@)????Mb??1D7?ި?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???Mb??!?̱?@)???Mb??1?̱?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?,?8?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	MX??????i+v???8gDio??!!?rh????	!       "	!       *	!       2$	??SK????NE??_p??? ?	???!,e?X??:	!       B	!       J$	ϛ\?t+??C?+2g>??;?O??n??!?X?Ѱ?R	!       Z$	ϛ\?t+??C?+2g>??;?O??n??!?X?Ѱ?JCPU_ONLYY?,?8?@b 