?($	:???@????Z?ѵ?????(\???!?ZB>????$	??(???@?kb?C?@,?D?@?@!O?3=6@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???&S??,Ԛ????AW?/?'??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:M??????B?i??ANё\?C??Y??ͪ?զ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????9#??[Ӽ???AR'??????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ZB>????=?U????Avq?-??Y?u?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????_v????ܵ???A?&1???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??@????8??d?`??A[????<??Y]m???{??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??n????U???N@??AjM??St??Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z??????Mb??A????z??Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&<Nё\??????????A?4?8EG??Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	p_?Q??a??+e??A?:pΈ???Y ?o_Ι?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?v??/??????_v??A??Q???Y6?;Nё??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ?????o_??ANё\?C??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k+??ݓ??o??ʡ??A??H.???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY?????1w-!??A???1????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?]K?=????q????A?%䃞???Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??h o???ŏ1w-??A??Q???Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?_vO?????o_??A~8gDi??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H?}????6???A.???1???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??{??P???HP???Au????Y?|a2U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Q?|a2????H?}??A????Y䃞ͪϥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(\?????H?}??Aݵ?|г??Y㥛? ???*	43333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???H??!E?w?}@@)n4??@???1+!?>@:Preprocessing2F
Iterator::Model?St$????!S?b}??C@);?O??n??1C??Yu	9@:Preprocessing2U
Iterator::Model::ParallelMapV2??H.?!??!?L?A??,@)??H.?!??1?L?A??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???_vO??!?n??NN@)P??n???1?|?u?m&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?/L?
F??!|H)O??@)?/L?
F??1|H)O??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?4?8EG??!~/???(@)Tt$?????1??^??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??0?*??!{?b??i0@)Zd;?O???1??>??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??e?c]??!??p߸C@)??e?c]??1??p߸C@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 56.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??"? @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?-7׮???˝#??,Ԛ????!=?U????	!       "	!       *	!       2$	??ǲ?e?????V?t??ݵ?|г??!vq?-??:	!       B	!       J$	??Y?Ln???YUgI?????????!?v??/??R	!       Z$	??Y?Ln???YUgI?????????!?v??/??JCPU_ONLYY??"? @b Y      Y@qo???gR@"?
both?Your program is POTENTIALLY input-bound because 56.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?73.6208% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 