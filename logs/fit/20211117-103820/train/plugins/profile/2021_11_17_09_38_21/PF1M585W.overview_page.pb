?($	???k#???3XG ?:????ܵ???!$(~??k??$	?K??@????@??$E?@!@r?yW7)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?l??????_?L?J??A?D?????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~??k??}?5^?I??A[B>?٬??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C?l????jM??St??A????o??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?6?[ ???~?:p???A????H??YHP?sע?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ?????h o???A?<,Ԛ???Y??ܥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Z??ڊ?????k	????A?&S???YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;???R'??????AY?8??m??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Mb???c]?F??A؁sF????Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&mV}??b???*??	??Aa??+e??Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	Y?8??m???u?????A?[ A?c??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??j+????????????A?6?[ ??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R??x$(~??A?t?V??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=?U?????K?46??A c?ZB>??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?? ?????C?l???AǺ????Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??????q=
ףp??A?T???N??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.???uq???AM?O???Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?2ı.n?????B?i??A}гY????YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Q???Q?|a2??A?]K?=??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?=?U??bX9????A?4?8EG??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O???TR'?????A?):????Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?????ܵ?A2w-!???Y??+e???*	????̌?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{?/L?
??!??eC?B@)?"??~j??1???A@:Preprocessing2F
Iterator::Model?H?}??!px?۴@@)"??u????1~??ZOB2@:Preprocessing2U
Iterator::Model::ParallelMapV2?46<??!??V??N.@)?46<??1??V??N.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???߾??!??C5??P@)?.n????18?="'?&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatel	??g???!?_3?+@)6?;Nё??1fue*?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?E???Ԩ?!cƥ?;?@)?E???Ԩ?1cƥ?;?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapH?z?G??!Ѐ%???1@)????o??1y??ŌK@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?
F%u??!?????
@)?
F%u??1?????
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??41TQ@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?彘?e????	?????ܵ?!}?5^?I??	!       "	!       *	!       2$	{.}????*-??߶?2w-!???!????o??:	!       B	!       J$	??@??ǘ?~?b?,????!??u???!B>?٬???R	!       Z$	??@??ǘ?~?b?,????!??u???!B>?٬???JCPU_ONLYY??41TQ@b Y      Y@q????ScI@"?
both?Your program is POTENTIALLY input-bound because 49.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?50.776% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 