?($	贁N???PD?\}????6?[??!d;?O????$	??dk.@°?H@9*
S?? @!V?Z??=,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??W?2????n??ʱ?A??%䃞??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??A?f????a??4??A|a2U0??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"?uq???8EGr???Ah??|?5??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????K????^)??AKY?8????Y??ݓ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d;?O????F%u???A?H?}??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?K7?A`???Ǻ????AL?
F%u??Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ꕲq?????ݓ????A??a??4??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???N@??t$???~??A??MbX??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?5?;N????j+????A      ??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	W[????????6?[??A?:pΈ???YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??A?f???E??????A?-????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???V?/???=yX???A?:pΈ???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z????	h"??A؁sF????YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$(~?????W[?????A??T?????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????_?L??A?c?ZB??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n????c]?F??A?f??j+??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2w-!?????St$???Aq=
ףp??Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&l	??g????sF????AB>?٬???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;M?O??46<?R??A????z??Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ??A?c?]K??A??z6???YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??6?[??H?z?G??A?i?q????Y)\???(??*	?????g?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ڊ?e??!?fA??A@)f??a????1?#EC?q@@:Preprocessing2F
Iterator::Model??s????!?՞???@)q???h??18R4??2@:Preprocessing2U
Iterator::Model::ParallelMapV2M??St$??!?fA??d*@)M??St$??1?fA??d*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}??b???!??J?Q@)??JY?8??1?='?W)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?w??#???!??d?*a @)?w??#???1??d?*a @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate5^?I??!?L_%??.@)?D???J??1???U??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?(??0??!:??J?3@)??H?}??1??;?H?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*{?G?z??!)??[@){?G?z??1)??[@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?? 1??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	`zf2??????~?޼?H?z?G??!?8EGr???	!       "	!       *	!       2$	?rs?ʬ???R9?'????i?q????!??MbX??:	!       B	!       J$	;ɗ?7????	?XM?}????߾??!n????R	!       Z$	;ɗ?7????	?XM?}????߾??!n????JCPU_ONLYY?? 1??@b Y      Y@qH~
?J@"?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?52.1527% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 