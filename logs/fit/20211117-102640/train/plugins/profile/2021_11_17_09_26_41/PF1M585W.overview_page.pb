?($	?L֑ϖ??W?( ????k+??ݓ??!/n????$	????!@?j)@?RL?]%@!w?\&65@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$a2U0*?????ڊ?e??A?'????Y?[ A???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?o_???/n????AC??6??Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ё\?C???q???h ??A?Fx$??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;pΈ?????-????A?I+???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??d?`T???	h"lx??A^K?=???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ݓ??Z??Gx$(??A?QI??&??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?k	??g??]?Fx??A'1?Z??Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/n?????9#J{???A?uq???YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m???????A㥛? ???Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ܵ?|???H?}8??A??:M???Y?U???؟?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
'?W????MbX9??A?0?*???Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z?):????B>?٬???A???H.??Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e?c]???l	??g???A|a2U0??Y??镲??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??K7?A??
ףp=
??A??\m????Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0?'????W?2??A\???(\??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[??*??D???A?|a2U??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h ????j+????A'?W???Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e?c]?????ͪ????Au?V??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Tt$??????^)???A c?ZB>??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?>W[?????H.?!???A_)?Ǻ??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ??Έ?????A~8gDi??Y?]K?=??*	?????N?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??H.???!?HHY(w@@)??x?&1??1?JI?:>@:Preprocessing2F
Iterator::Model-!?lV??!?-XqW?B@)^?I+??1????T7@:Preprocessing2U
Iterator::Model::ParallelMapV2????ׁ??!?35?+4-@)????ׁ??1?35?+4-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_)?Ǻ??!Hҧ??O@)?"??~j??1???->E%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?D?????!9~??:a,@)?ݓ??Z??1?W?M?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???(\???!???'?@)???(\???1???'?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??m4????!֔?Sa?2@)
ףp=
??1?V??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?!??u???!N6?K??@)?!??u???1N6?K??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9`?;ÐC@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	iK?8n?????z???Έ?????!?9#J{???	!       "	!       *	!       2$	]S?/?X??	?+P/??~8gDi??!㥛? ???:	!       B	!       J$	????}\????4?%???~j?t???!?[ A???R	!       Z$	????}\????4?%???~j?t???!?[ A???JCPU_ONLYY`?;ÐC@b Y      Y@q??P?uC@"?
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.0348% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 