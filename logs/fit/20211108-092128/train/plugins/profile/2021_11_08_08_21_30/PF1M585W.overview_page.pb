?($	a??#?????f????Nё\?C??!f?c]?F??$	z~'P2@>HAq?.@?H?,?
??!??'?\2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??e?c]?? ?o_ι?A0*??D??Y?*??	??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~?:p????;Nё\??Af??a????YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$???~???L?
F%u??A-C??6??Y?lV}???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%u???}??b???AX?2ı.??YjM??S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F??lxz?,C??Ae?`TR'??Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[Ӽ????W?2??A]?Fx??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|?5^?????1??%??Ao???T???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș?????[????<??A@a??+??YDio??ɤ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^?I+??㥛? ???Ax$(~??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	o??ʡ??^?I+??A?J?4??Y?@??ǘ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?	????Q???A?8??m4??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|гY??????z??Al	??g???YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|??Pk?????ʡE??A??A?f??Y???JY???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??q=
ףp??A??V?/???Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B??8gDio??A???(\???Y_)?Ǻ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ڬ?\m???4??7????A????z??Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????A?f????A????z??Y䃞ͪϥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?/L?
??]m???{??AHP?s??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C?i?q?????Pk?w??AQ?|a??Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F??ޓ??Z???AO??e???Y?V-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C??????x???A?[ A???Y??#?????*	33333!?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ݓ??Z??!?a??v?@@)F????x??1:????>@:Preprocessing2F
Iterator::Modelc?=yX??!??=Y?C@)(~??k	??1??w??9@:Preprocessing2U
Iterator::Model::ParallelMapV2;M?O??!??uE,@);M?O??1??uE,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z?G???!z8¦?,N@)0*??D??1V???>?#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?Fx$??!Fy?eG@)?Fx$??1Fy?eG@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate@a??+??!??;2`?)@)?߾?3??1?wd?Z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%??C???!hU2DO1@)??H?}??1idQ P?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*%u???!+????A@)%u???1+????A@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??ptR@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	U?!?X??Jб?4	??????x???!ޓ??Z???	!       "	!       *	!       2$	?	?^????????F???[ A???!?J?4??:	!       B	!       J$	rizLc????3(n?q??X9??v???!?*??	??R	!       Z$	rizLc????3(n?q??X9??v???!?*??	??JCPU_ONLYY??ptR@b Y      Y@q?Z
???8@"?
both?Your program is POTENTIALLY input-bound because 55.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?24.6858% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 