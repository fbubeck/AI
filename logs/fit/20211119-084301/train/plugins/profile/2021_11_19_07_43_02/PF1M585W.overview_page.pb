?($	???}????c?{???_?L??!??9#J{??$	i?#+j2@?I?w?@5?LoP@!?r??.7@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$A?c?]K???_vO??A???H??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)??0?????H?}??A0L?
F%??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??D?????:pΈ???A??ǘ????Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ?|????Q???A)?Ǻ???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"?uq??\ A?c???A?C??????Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??z6???f??a????A?+e?X??Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*:??H??V}??b??A?rh??|??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??{??P???n?????A??St$???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ǘ?????T㥛? ??A???3???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	333333???_?L??A?MbX9??Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?\?C?????w??#???Ac?=yX??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J{?/L????lV}???A??ZӼ???Y???S㥋?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*?????%䃞??A?q??????Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_)?Ǻ??jM????A)\???(??Y_?Qڋ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?]K???????Mb??A|a2U0??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?46<???46<??A?A?f???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{??!?lV}??A??T?????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v???3??7??AR'??????Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?	?????ݓ????Affffff??Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e?c]???'????A?Y??ڊ??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_?L??F%u???A???(\???Y??_vO??*	      ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatk+??ݓ??!7??Mo?@@)ˡE?????1!Y?B?>@:Preprocessing2F
Iterator::Modelk?w??#??!???,d1B@)?.n????1Y?B7@:Preprocessing2U
Iterator::Model::ParallelMapV2J+???!???,d?*@)J+???1???,d?*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipI.?!????!Nozӛ?O@)؁sF????1     `(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceȘ?????!???7??@)Ș?????1???7??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate1?*?Թ?!??,d!k+@)????????1B???,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;pΈ????!-d!Y?1@)??????1??,d!?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*M?O???!pzӛ??@)M?O???1pzӛ??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9i??GՈ@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?Y<???ܧ????F%u???!!?lV}??	!       "	!       *	!       2$	PdU???????4?????(\???!??T?????:	!       B	!       J$	??N?8???d?a?֐????S㥋?!?I+???R	!       Z$	??N?8???d?a?֐????S㥋?!?I+???JCPU_ONLYYi??GՈ@b Y      Y@q??!??H@"?
both?Your program is POTENTIALLY input-bound because 51.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?48.1926% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 