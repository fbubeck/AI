?($	??ݩ?@??]???F???k	??g??!7?[ A??$	?ID?^F@#N?݄@ʭ6χ?@!?ҋC.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?k	??g??^?I+??AO??e?c??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A?f???V-???A?5^?I??Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?镲q????(???A?E??????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&S??????QI??A???S????Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio???i?q????A???Q???Y2??%䃎?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P?s????0?*??A?=yX?5??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????M??l	??g???A??????Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A?????(????A)\???(??Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?N@a???J+???A?????Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	x??#????RI??&???A[????<??Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
?????ڊ?e??A?c?ZB??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&gDio?????(\?????A????????Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z???<Nё\???A?_vO??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q???<,Ԛ???A?&S???Y	?^)˰?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?[ A??=
ףp=??A[????<??Y??K7?A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B>?٬????HP???A??#?????Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?JY?8??? A?c?]??A??a??4??Y??6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???B?i??'1?Z??At$???~??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4??m???????A???߾??Y?MbX9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=?U????46<?R??A?(??0??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??S㥛????K7?A??A$(~????Y??D????*	?????%?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?/?$??!w??3?A@),e?X??1?A+???@:Preprocessing2F
Iterator::ModelyX?5?;??!*????B@)?a??4???1 Q???4@:Preprocessing2U
Iterator::Model::ParallelMapV2O??e???!?{61q\0@)O??e???1?{61q\0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_)?Ǻ??!??;l`O@)?~?:pθ?1?BD?$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??????!Yܜ?9?*@)Q?|a??1$?0???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?@??ǘ??!????@)?@??ǘ??1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Pk?w??!^?@??$1@)???????1?????U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?W[?????!???>?	@)?W[?????1???>?	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??b??9@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?Z?t?t??X:N0???^?I+??!?<,Ԛ???	!       "	!       *	!       2$	?.?lK!??ͽ?????O??e?c??![????<??:	!       B	!       J$	۠? W????.?6???ZӼ???!	?^)˰?R	!       Z$	۠? W????.?6???ZӼ???!	?^)˰?JCPU_ONLYY??b??9@b Y      Y@qF?T?T@"?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?82.3802% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 