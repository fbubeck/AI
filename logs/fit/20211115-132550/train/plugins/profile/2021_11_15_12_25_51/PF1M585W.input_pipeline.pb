$	?????????????`TR'???!????z??$	=O?S?N@*'ѝ?}@.???% @!dN?О.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!?lV}???0?*??A??q????Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?!?uq???<,Ԛ???A5^?I??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?	???:#J{?/??A?ZB>????YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?????S㥛??A?HP???Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???|??Pk???A?H.?!???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C??6??????_v??A??~j?t??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2??Ӽ????A???&S??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE???D????9??A????_v??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S?!?uq???-???1??A~??k	???YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	6<?R?!???ʡE????AU0*????Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?Ǻ??????ZӼ???A???????Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???3???Ǻ?????A7?[ A??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{?? o?ŏ??A??x?&1??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&S????"??~??A??4?8E??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z??C??6??A??{??P??Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n??????1????A:??H???YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Q??????????A?&?W??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??n????4??@????A???????Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h???O??e??A'?W???Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ׁs??????_v??A???9#J??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?`TR'?????m4????A+??ݓ???Y??g??s??*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S????!mz]b??B@)%u???1??t!OA@:Preprocessing2F
Iterator::Model5?8EGr??!u???̭@@)?ZB>????1?^dm?3@:Preprocessing2U
Iterator::Model::ParallelMapV2?_?L??!^?Ƥ?*@)?_?L??1^?Ƥ?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?٬?\m??!???P@)䃞ͪϵ?1;+E?$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?V?/?'??!?v???-@)      ??1C:Þ?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???_vO??!??Q?@)???_vO??1??Q?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????ׁ??!????'?2@)????o??1{]b??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*=?U?????!?F׾?@)=?U?????1?F׾?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?U???B@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	<v*?i(??? G??????m4????!C??6??	!       "	!       *	!       2$	:ʀ1d???ݵ!^???+??ݓ???!?HP???:	!       B	!       J$	t?@?t???j??????V-???!??q????R	!       Z$	t?@?t???j??????V-???!??q????JCPU_ONLYY?U???B@b 