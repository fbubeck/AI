?($	??)n0i??b ??sF?????h o??!V????_??$	??4|??@)}?3?G
@??1????!p ?z?1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??y??????1段?A?B?i?q??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&?W??Q?|a2??A?c?ZB??Ya2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???1?????ZӼ???A?ڊ?e???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX??????ׁs??A??ڊ?e??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9??o??ʡ??AHP?s??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V????_????9#J{??AM?J???Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a???A`??"??A9EGr???Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c]?F???5?;N???A??(????YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&3ı.n????lV}???A??^)??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ݓ??????C?l???A????_v??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
J{?/L???`vOj??A
h"lxz??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?>W[?????8??m4??A+?????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ӽ?????w??#???A???h o??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&5^?I???=yX?5??A&䃞ͪ??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?c???ǘ?????A??ʡE???Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o??ʡ????1??%??A6<?R?!??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.???~j?t???A???<,???Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|?5^???:#J{?/??AE???JY??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ԛ??????~?:p???A?x?&1??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l???q???h??Affffff??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???h o??*??D???A?8EGr???Y???H??*	????̌?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\???(\??!???Tp?A@)??(???1????@@:Preprocessing2F
Iterator::Model???镲??!0???d?>@)U???N@??1]?ӳ1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipf??a????!t?NϦRQ@)??????1L;k?-@:Preprocessing2U
Iterator::Model::ParallelMapV2}?5^?I??!D<?:"*@)}?5^?I??1D<?:"*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicee?X???!?X??>J @)e?X???1?X??>J @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenateŏ1w-!??!?Tpm?,@)???<,Ԫ?1I?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapU0*????!??^o#3@)ˡE?????1`?*??M@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??ZӼ???!?S???5@)??ZӼ???1?S???5@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??n\(?
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	:ʀ1d???Q??^??????1段?!??9#J{??	!       "	!       *	!       2$	?Q?`E??????*?????8EGr???!M?J???:	!       B	!       J$	ϱ?????:??z/D????H?}??!J+???R	!       Z$	ϱ?????:??z/D????H?}??!J+???JCPU_ONLYY??n\(?
@b Y      Y@qn\?4;A@"?
both?Your program is POTENTIALLY input-bound because 50.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.4625% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 