?)$	??xv????yEl&????v??/??!?-????$	?Wխg@>Ny3?L@?2??:X @!??#j?1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?v??/??o???T???AǺ?????YK?46??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&f?c]?F?????Mb??A?[ A???Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,Ԛ?????ڊ?e???Ap_?Q??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????o????JY?8??A#J{?/L??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZB>????+??ݓ???A?Q?????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&3ı.n???$(~????A?=yX?5??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??u?????O??n??A?>W[????Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??/?$??,e?X??A	?c???Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C??????8??d?`??A????Mb??Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??K7?A?????T????A???JY???Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?St$?????c?ZB??AX?5?;N??Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I??aTR'????A6<?R?!??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??St$?????k	????A?u?????Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?l???????a??4???A9??m4???Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?-????]?Fx??A??a??4??YǺ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Gx$(??_?Q???Aı.n???Y1?*????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???&S???HP???A?A?f???Y??y?)??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J???	?c???A8gDio???Y??MbX??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z??q???h ??Ai o????YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?T???N??????Q??Aŏ1w-??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%u???؁sF????AL?
F%u??Y46<?R??*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????!?5eMY?@@)???镲??15eMYSV>@:Preprocessing2F
Iterator::Model#??~j???!5eMYS?B@)????S??1?5eMY?8@:Preprocessing2U
Iterator::Model::ParallelMapV2}?5^?I??!?)kʚ?)@)}?5^?I??1?)kʚ?)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???x?&??!˚???)O@)Ԛ?????1?????&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?c?ZB??!_??}+@)T㥛? ??1T֔5eM@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicelxz?,C??!kʚ???@)lxz?,C??1kʚ???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*???!w?qG?1@)r??????1qG?w@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??@??ǘ?!?/???@)??@??ǘ?1?/???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t49.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9̜\j?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	v??4?????>ʫ???o???T???!????Q??	!       "	!       *	!       2$	Vg??????W??L?
F%u??!??a??4??:	!       B	!       J$	?[Tߎ??????????2??%䃎?!1?*????R	!       Z$	?[Tߎ??????????2??%䃎?!1?*????JCPU_ONLYY̜\j?@b Y      Y@q?J!??>@"?	
both?Your program is MODERATELY input-bound because 5.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t49.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?30.9798% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 