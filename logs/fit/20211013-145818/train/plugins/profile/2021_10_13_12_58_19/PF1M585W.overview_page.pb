?($	p?????4?G9????~j?t??!ݵ?|г??$	?.X?@Z?҈?@nh??ц@!?c???+-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$o??ʡ??=,Ԛ???Aa??+e??Y??b?=??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a2U0*????Zd;???A o?ŏ??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0L?
F%???[ A???A?ڊ?e???Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx?????B?i??A?T???N??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n????
h"lxz??A??ʡE???Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾???*:??H??Ak+??ݓ??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????JY?8??A?:pΈ??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C????A?f??A?!?uq??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e??a????W?2???A??a??4??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	ݵ?|г???/?$??AY?8??m??Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
L?
F%u???A?f???A???????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??o_??h??|?5??A?9#J{???Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?X????????o??A~??k	???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??z6????0?*???A?=yX?5??Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??y???jM??St??A?MbX9??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G????:pΈ??Ac?ZB>???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ??m???{???A??ZӼ???Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&C?i?q???tF??_??As??A??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&io???T??c?ZB>???A?^)???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF?????C?l????A?W?2ı??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??~j?t??????????Affffff??YV-???*	fffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??????!KB?E??@@)??(\????1??^??>@:Preprocessing2F
Iterator::Model??_?L??!=?\??A@)I.?!????1??:???6@:Preprocessing2U
Iterator::Model::ParallelMapV2?}8gD??!?î?/*@)?}8gD??1?î?/*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?l??????!b??Q	P@)??u????1?[??%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceV-???!?)T?@)V-???1?)T?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?D?????!?	?=?+@)?3??7??1????x@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"lxz?,??!???e?3@)bX9?Ȧ?1?hS??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?ݓ??Z??!!O?e?@)?ݓ??Z??1!O?e?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?sq9?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	c?ZB>???U ????????????!?/?$??	!       "	!       *	!       2$	@???y???l阯/??ffffff??!Y?8??m??:	!       B	!       J$	?U\?Ә?_?????V-???!???_vO??R	!       Z$	?U\?Ә?_?????V-???!???_vO??JCPU_ONLYY?sq9?@b Y      Y@qcnc?U@"?
both?Your program is POTENTIALLY input-bound because 48.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?86.9904% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 