?($	H?`?q???%3???????????B??!?ʡE????$	M?3$?@!??a?t	@??L???!???b1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$%??C?????e??a??A7?[ A??Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ʡE?????	h"lx??A??e?c]??Y?HP???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)????ZB>????A?c]?F??Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZB>????F%u???Ak+??ݓ??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s????2??%????A?q??????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?U??????~8gDi??A???<,???Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h ???ZӼ???AV-????Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V????ܵ??A9??v????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t$???~??9??m4???A5?8EGr??Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???&S??h??|?5??A???V?/??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??3????	?c?Z??A???????YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??+??ݓ???A??6?[??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& c?ZB>??q???h??Alxz?,C??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????_v????ǘ????A?1w-!??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&^K?=????	???AHP?s???Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?c?Z??'1?Z??AX?2ı.??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx??+??	h??A$???~???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&yX?5?;??x$(~???AF%u???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a??+e??#J{?/L??A?H?}8??Y??g??s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ??W[??????A??/?$??Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B?? ?~?:p??A?o_???Y"??u????*????̴?@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!??k
E@)??????1RL?0K?C@:Preprocessing2F
Iterator::ModelB>?٬???!A????=A@)?4?8EG??1?LR?͘5@:Preprocessing2U
Iterator::Model::ParallelMapV2\???(\??!?:*??)@)\???(\??1?:*??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?\?C????!?*??aP@)??ڊ?e??1????:#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice&S????!??? ??@)&S????1??? ??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatex$(~???!+?c?L?$@)???JY???1?
?_?'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?X????!??mL?+@)vq?-??1?i???
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??0?*??!x]?s?@)??0?*??1x]?s?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 54.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9v?/?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	e)>??1???|? ??????e??a??!?	h"lx??	!       "	!       *	!       2$	T?fW?t????ǻ]???o_???!X?2ı.??:	!       B	!       J$	z}????????淪???<,Ԛ???!ŏ1w-!??R	!       Z$	z}????????淪???<,Ԛ???!ŏ1w-!??JCPU_ONLYYv?/?@b Y      Y@qzY?D&@@"?
both?Your program is POTENTIALLY input-bound because 54.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.299% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 