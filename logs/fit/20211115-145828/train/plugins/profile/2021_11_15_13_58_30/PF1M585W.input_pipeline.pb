$	?Mq?I??>L?*???a2U0*???!u?V??$	XVj?"s@d?ϩ??@?p??A@!3lm<??1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Mb??ffffff??A?JY?8???Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f??a????????9#??A?R?!?u??Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?V????|гY??A?V?/?'??Y-!?lV??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v????*??	??A6?>W[???Y?'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y?)?? ?~?:p??A??0?*??Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&HP?s??;?O??n??Ao???T???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4?????_vO??A??	h"l??Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??S???=?U???Aq???h ??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'???????P?s???A}?5^?I??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???9#J??>yX?5???A	?^)???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??H.?????u????AU0*????YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?QI??&??"?uq??A?f??j+??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\??&S??:??A{?G?z??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??ݓ???ˡE?????A;M?O??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?+e?X??\ A?c???A`??"????Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-???1??&S??:??A???????YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)?Ǻ????Y??ڊ??A??|?5^??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?rh??|??????߾??A}??b???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?h o?????H.?!??A?鷯??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?,C?????_vO??A??ܵ???Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*???c?ZB>???At??????Y?
F%u??*	23333O?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatC?i?q???!xSM{?4A@)?6?[ ??1?????@:Preprocessing2F
Iterator::Model?lV}???!}?t5?UA@)V????_??1S?07?4@:Preprocessing2U
Iterator::Model::ParallelMapV2ё\?C???!LE?tj?+@)ё\?C???1LE?tj?+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????????!B?E?UP@)?z6?>??1??si??%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????߾?!?oΉD?,@)r??????1??؃?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceV}??b??!?q?(?@)V}??b??1?q?(?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapV-???!T??i"4@)(~??k	??1Fj??;@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?e??a???!D??? @)?e??a???1D??? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???g?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???6 ???PflGi;??c?ZB>???!??|гY??	!       "	!       *	!       2$	Ωs?W???/a??[??t??????!?V?/?'??:	!       B	!       J$	?GǫE???b?vZ??lxz?,C??!-!?lV??R	!       Z$	?GǫE???b?vZ??lxz?,C??!-!?lV??JCPU_ONLYY???g?@b 