$	?֍??????mr^?a????(\????!??ܵ???$	5Y?@?v??@?M??Xe@!/9?x4?+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?q??????鷯??A????Mb??Y?????K??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??{??P????JY?8??A??(???Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?^)???V-???A@a??+??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h??|?5????"??~??A??ZӼ???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*???F??_???AX?5?;N??YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]m???{??<Nё\???A|??Pk???Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?$??C??2??%????A?u?????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???o_????#?????A????(??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Q??????o_??AV-????Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?Y??ڊ????ܵ??A???ׁs??YQ?|a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??#????????1????AS??:??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[B>?٬???e??a???AD?l?????Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?????0?*??A1?*????Y?-?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P?s???w??/???A??y?):??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ?????7??d??Aj?t???YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???s??A???AU???N@??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C?l??????_vO??Ad]?Fx??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???S???
ףp=
??A???<,???Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????KY?8????A?_?L??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ???????S???A?Zd;???Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(\?????Zd;߿?Ax$(~???Y???H??*	??????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^K?=???!?@`???A@)Ǻ????1????D@@:Preprocessing2F
Iterator::Model??|гY??!??u?Z?A@)?ZB>????1?A??*5@:Preprocessing2U
Iterator::Model::ParallelMapV2?\m?????!??F?f,@)?\m?????1??F?f,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipEGr????!?E??1P@)0*??D??1#J??
?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??(???!???Ya?@)??(???1???Ya?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateȘ?????!?0b?s*@)?J?4??1??!j?e@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?b?=y??!p+?^??0@)??q????13? l}?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?D???J??!??Dw?	@)?D???J??1??Dw?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???}Q@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	Ɍv???!zX?????鷯??!??ܵ??	!       "	!       *	!       2$	E?f??9??΅???4??x$(~???!1?*????:	!       B	!       J$	&?l?L???65??????H??!?????K??R	!       Z$	&?l?L???65??????H??!?????K??JCPU_ONLYY???}Q@b 