?($	?7z?n?????)??m???????!?????B??$	\?ŵf?@N)D<@(?U?@!Ӄ??}(.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$h??|?5??TR'?????A?G?z??YI.?!????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z???D?????A~8gDi??Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[B>?٬??-!?lV??AyX?5?;??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7?????z?G???Aq=
ףp??YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??a??4???MbX9??A?ZB>????Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????M??,e?X??A?G?z???YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??\m????aTR'????Ar?鷯??Ylxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????B??(??y??A??C?l??Y??(\?¥?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ffffff??r??????A??镲??Y?
F%u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	;?O??n??gDio????AU0*????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??d?`T??c?ZB>???A?h o???Y??6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?^)?????y?)??Aj?t???Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F????x?????B?i??A6?;Nё??Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?D??????ZB>????A-C??6??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+?????????B?i??AR???Q??YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\???9#J{???A??y???YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???<,???i o????A?):????Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZB>????U0*????Az?):????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4????V-?????A'1?Z??Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@?߾????????A?]K?=??Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&m????????? ?rh??A"?uq??Y???????*	?????4?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?X????!
?Jw)B@)?2ı.n??1*?M?m?@@:Preprocessing2F
Iterator::ModelM?O????!??&X.o>@)??JY?8??1????;3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??????!T[?i4dQ@)??C?l???1?6?'(@:Preprocessing2U
Iterator::Model::ParallelMapV2R'??????!U??0?f&@)R'??????1U??0?f&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?MbX9??!??p?F?1@)??yǹ?1u?d?5P&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-!?lV??!?"?دd@)-!?lV??1?"?دd@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??@?????!???J?r5@)??y?):??1?w!r??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??JY?8??!????;@)??JY?8??1????;@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?,!q`?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	2F?8?R??*P䯷???TR'?????!???B?i??	!       "	!       *	!       2$	?
F%u??qg&U???"?uq??!??C?l??:	!       B	!       J$	??????	?7H??	?^)ː?!?:pΈ??R	!       Z$	??????	?7H??	?^)ː?!?:pΈ??JCPU_ONLYY?,!q`?@b Y      Y@q??U??U@"?
both?Your program is POTENTIALLY input-bound because 46.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.1341% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 