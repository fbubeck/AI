?($	?t????}#?=???????z6??!?/?$??$	?i?r??@??[???@gɣ¹@!c??$})@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???????=
ףp=??A???_vO??Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<????W?2??A???H.??Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??C?l????i?q????A$(~??k??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&~8gDi??n4??@???A????????Y?N@aã?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&p_?Q??@?߾???A6<?R?!??YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t$???~??F%u???AjM????Y?5?;Nѡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?$???:pΈ???A??y?):??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??V?/???8gDio??A?0?*???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|?5^??+??????A~8gDi??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??S㥛???[ A?c??Aꕲq???Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?}8gD????|гY??A"?uq??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e??a???F??_???Ak+??ݓ??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/L?
F???X????AtF??_??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c]?F??V-????A?:M???Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\??L7?A`???A46<???Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[Ӽ?????(???A?V-??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???C??6??A?z?G???Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O?????h o???A4??7????Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&i o????L7?A`???A;pΈ????Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Tt$????????????A?6?[ ??YP?s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???z6??+??ݓ???A*:??H??Y???????*	gffff?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat c?ZB>??!I?<B@)RI??&???1"o?P??@@:Preprocessing2F
Iterator::Model???B?i??!?]=?@@)???(???1a??V>?3@:Preprocessing2U
Iterator::Model::ParallelMapV2\ A?c̽?!???G??*@)\ A?c̽?1???G??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_?L???!s3Qa??P@)EGr????1??z]??%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???????!u%k?? @)???????1u%k?? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??	h"l??!?????-@)??e?c]??1?n?r??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap:??H???!@?Z|??3@)??_?L??1qs?Y?/@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???????!M???Uf@)???????1M???Uf@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ǘ???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??q??d?8?L??+??ݓ???!???????	!       "	!       *	!       2$	/?!?^]??ƭ{??????_vO??!??y?):??:	!       B	!       J$	)݌B???????????Mb??!??e?c]??R	!       Z$	)݌B???????????Mb??!??e?c]??JCPU_ONLYYǘ???@b Y      Y@qP?????A@"?
both?Your program is POTENTIALLY input-bound because 49.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?35.8103% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 