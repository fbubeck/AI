$	%:3g??????^???!??u???!?H.?!???$	z=zk=?@?CY?g?@?7? t@!ܶm۶1/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Ǻ??????G?z??A?G?z???YW[??재?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H.?!????%䃞???AΪ??V???Y?k	??g??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"l??0*??D??Ah??s???Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;M?O???q??????A??? ?r??Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`??"????B?f??j??Aa??+e??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?C???????(\????A???1????Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z??O??e?c??A?????B??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W?/?'???~j?t???AP??n???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e??a???.?!??u??A?rh??|??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?5?;N????o_???A?v??/??Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?ܵ?|?????Pk?w??A?y?):???Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????(????ׁsF??A???{????Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n????a??+e??A?0?*???Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"lxz?,??=?U????AȘ?????Y??D????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??HP???s?????AC??6??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W[??????y?&1???A?e??a???Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&s??A????[ A?c??A??z6???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S?!?uq???ܵ?|???A???S????Y1?Zd??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???QI??e?X???A^?I+??YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??D????8??d?`??A c?ZB>??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?!??u?????ׁsF??A=?U?????Y??ܵ?|??*	gffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeath"lxz???!?p??JB@)F%u???1?S3???@@:Preprocessing2F
Iterator::Model?j+?????!???a??A@)Ϊ??V???1???15@:Preprocessing2U
Iterator::Model::ParallelMapV2V-????!?3?z?,@)V-????1?3?z?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????H??!?O'3P@)?~?:pθ?1?jPO?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceaTR'????!???M@)aTR'????1???M@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?ZB>?ټ?!?????)@)?-????1aAj??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapS?!?uq??!.???91@)n????1?/#?$?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor* ?o_Ι?!??Qw??@) ?o_Ι?1??Qw??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?	?<??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	E????????K$?????ׁsF??!?%䃞???	!       "	!       *	!       2$	N??K????]ӟ??=?U?????!??z6???:	!       B	!       J$	fkf??~??????LQ????ܵ?|??!W[??재?R	!       Z$	fkf??~??????LQ????ܵ?|??!W[??재?JCPU_ONLYY?	?<??@b 