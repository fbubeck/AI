?($	???_????r?v?????c]?F??!??u????$	`?e?@i?????@dQ?N?@!?N?	*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???{????9EGr???A??h o???Y??e??a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??u????؁sF????A\???(\??YEGr????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G?????s????A?c?]K???Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???x?&????|?5^??AZd;?O???YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&鷯?????I+???ApΈ?????Y<?R?!???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????;pΈ????A??ǘ????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|?5^???B?i?q??AjM??S??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*??????h o???A?D???J??Y\ A?c̝?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE??6<?R?!??A?q?????Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	-!?lV??sh??|???A>yX?5???Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?:M????5?;N???A?z?G???Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d]?Fx??d;?O????Aq???h ??Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&R???Q???J?4??A?J?4??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-?????9EGr???AZd;?O??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&$?????????_?L??A?/?'??Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/??gDio????A??g??s??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?Zd???>W[????A???T????Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx???ŏ1w??AH?z?G??Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?????? ?r??A??(???Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??4?8E???e??a???A]m???{??Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c]?F???St$????A?lV}????YU???N@??*	33333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat5^?I??!??b?A@)-!?lV??1?Q?G7B@@:Preprocessing2F
Iterator::Model?8??m4??!ڛ?0~?>@)??T?????1?ƈ??@1@:Preprocessing2U
Iterator::Model::ParallelMapV2??W?2???!5?\3+@)??W?2???15?\3+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0?'???!
Y?s`MQ@)??0?*??1?-:??(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateGx$(??!?;?X?0@)?~j?t???1o[???"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???ZӼ??!?Vm9@)???ZӼ??1?Vm9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaph"lxz???!H?8???5@)?|гY???1"??V??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*}гY????!?թ?a]@)}гY????1?թ?a]@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?hN9H@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??y?Zk???&S????St$????!؁sF????	!       "	!       *	!       2$	???????8Ȃ????lV}????!\???(\??:	!       B	!       J$	???????׀n$߆??/?$???!??e??a??R	!       Z$	???????׀n$߆??/?$???!??e??a??JCPU_ONLYY?hN9H@b Y      Y@q9??wXT@"?
both?Your program is POTENTIALLY input-bound because 53.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?80.0991% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 