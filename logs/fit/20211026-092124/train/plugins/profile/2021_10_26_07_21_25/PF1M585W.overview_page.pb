?($	H???????ᒌ?????"?uq??!?3??7??$	?T? ??@(??G?w	@?;????!?u/?6?/@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$<Nё\???????Q??Ar?鷯??YTt$?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ݓ??????ǘ????A????B???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7??~??k	???A?"??~j??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???+????A??1??%??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46??+?????A^?I+??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????1w-!??A?s?????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a???{??Pk??Ar??????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?p=
ף??g??j+???A??e?c]??Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0*??D???q?????A0?'???Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?ʡE????_)?Ǻ??A?i?q????Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
f?c]?F???ܵ?|???A?<,Ԛ???Y???<,Ԋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\?????(???ANё\?C??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??@?????Q???A?v??/??Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q?|??{?G?z??A???(\???Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ǘ????2w-!???A???<,???Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8??d?`??I.?!????AaTR'????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?#????????Pk?w??Aq???h??Y??#?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_?L??`vOj??AΪ??V???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ???j?t???Az6?>W??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&??X9??v???A????Q??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"?uq??HP?s??A(~??k	??Y}гY????*	?????%?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatTt$?????!?c@h}B@)[Ӽ???1? Ei?@@:Preprocessing2F
Iterator::Model????Q??!???O?@@)f??a????1Ϻ?,;4@:Preprocessing2U
Iterator::Model::ParallelMapV2?@??ǘ??!?ԑh??*@)?@??ǘ??1?ԑh??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,e?X??!???
ؗP@)??C?l??1r.t̽&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?Y??ڊ??!?L?v?@)?Y??ڊ??1?L?v?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate:#J{?/??!??B?>]+@)???<,Ԫ?1d@h}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Q????!
|Y?2@)?g??s???1 c?]?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?z6?>??!ҋ?3B?@)?z6?>??1ҋ?3B?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9N,m?p?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?١?=???k#eb??HP?s??!~??k	???	!       "	!       *	!       2$	? ????s̚$??(~??k	??!q???h??:	!       B	!       J$	?x???D<J^?'?????<,Ԋ?!Tt$?????R	!       Z$	?x???D<J^?'?????<,Ԋ?!Tt$?????JCPU_ONLYYN,m?p?@b Y      Y@q??0?C@"?
both?Your program is POTENTIALLY input-bound because 53.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.1883% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 