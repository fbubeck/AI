?($	⍞?? ??ç?,???E???JY??! o?ŏ??$	C7?N?@???\?w@1bĈQ@!mU??N?6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?O??e???(??0??Ak?w??#??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C?i?q??????????Ax$(~???YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2ı???-????A ?~?:p??Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ??z6?>W??Aw-!?l??YгY??ں?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZB>???????ׁs??A??K7?A??Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-??????㥛? ???A{?G?z??Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????_v????ܵ?|??AO??e???Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?<,Ԛ????x?&1??A??/?$??YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U??F%u???A?c]?F??Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??6?[??m???????A?/L?
F??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
???S???S?!?uq??AK?=?U??Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????9EGr???A_?L???Y?X?? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??1??%???O??n??A??St$???Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5^?I???J?4??A????????Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}??b????W?2??A?D???J??YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n????%䃞???A??ׁsF??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??W?2???	?^)???A???QI???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????ܵ??A$???????Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e?c]???ŏ1w-??A??QI????Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&R???Q??>?٬?\??A??H?}??Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&E???JY???H.?!???A??????Y?? ?rh??*	33333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8gDio??!???sg?A@)?{??Pk??1%C?B@@:Preprocessing2F
Iterator::Model?t?V??!* f?B@)$(~??k??1?j??8@:Preprocessing2U
Iterator::Model::ParallelMapV21?*????!??ωx?)@)1?*????1??ωx?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipTR'?????!?????O@)S??:??1`?}?f?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicez6?>W[??!??۝??@)z6?>W[??1??۝??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate;pΈ????!??Am?2(@)????Mb??1?,?<W?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapmV}??b??!E?cu??0@)Ș?????1V?
?ް@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*T㥛? ??!<???U"@)T㥛? ??1<???U"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?q-[?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	1??????#x ?~???H.?!???!z6?>W??	!       "	!       *	!       2$	!???b????i+??????????!w-!?l??:	!       B	!       J$	@?߾????桨?????+e???!гY??ں?R	!       Z$	@?߾????桨?????+e???!гY??ں?JCPU_ONLYY?q-[?@b Y      Y@q??g?=C@"?
both?Your program is POTENTIALLY input-bound because 55.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.4811% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 