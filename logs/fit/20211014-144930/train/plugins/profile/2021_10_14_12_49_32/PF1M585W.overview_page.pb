?($	?̧? ??? mb2?S??]?C?????!o??ʡ@$	????X?@??lE=?@Y??M?@!?E???3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$H?z?G???W[?????A(??y??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ???c?ZB??A	?c???Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'??bX9????A'???????Y ?o_Ω?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??S㥛???A?f???A6?>W[???YF%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????B?i?q??A%u???Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[???H.?!???A???K7???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?h o????i?q????A?rh??|??Y??6???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O???Qk?w????A?MbX9??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?7??d????-???1??AI??&??YI.?!????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	o??ʡ@\???(\??Aŏ1w-??Y?=?U???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
P??n??????QI???A??y?):??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?n??????Q?|??AQ?|a??Yt$???~??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?e?c]?????HP??A?G?z??Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&yX?5?;??F%u???A?&1???Y0L?
F%??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??<,Ԛ??V-???AW[??????YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>?????MbX??A?e??a???Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?????(????A o?ŏ??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?'???????QI???Aw??/???Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????C?l???A????Mb??Y?;Nё\??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]???~?:p???A??+e???Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?C?????n????A?Zd;???Y|??Pk???*	?????3?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???{????!s?G???A@)-C??6??1?e??@:Preprocessing2F
Iterator::Model&䃞ͪ??!???)?B@)Gx$(??1?""D?P7@:Preprocessing2U
Iterator::Model::ParallelMapV2	?c?Z??!??-???+@)	?c?Z??1??-???+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip}??b???!@s?w?hO@)?? ?rh??1?
??VQ"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!-???),@)S?!?uq??17?'??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?_vO??!??n?s@)?_vO??1??n?s@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"??u????!Q?i??2@)]?C?????1+ONP?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??MbX??!̸!\ ?
@)??MbX??1̸!\ ?
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???hV@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	h?]e??????4?g,???W[?????!???QI???	!       "	!       *	!       2$	?0?*????,?>=??(??y??!ŏ1w-??:	!       B	!       J$	?b?5???a C?2???R?!?u??!?=?U???R	!       Z$	?b?5???a C?2???R?!?u??!?=?U???JCPU_ONLYY???hV@b Y      Y@q?^???:R@"?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?72.9215% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 